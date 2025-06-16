from datetime import datetime, timedelta
from functools import lru_cache
import logging
import os
import pytz
import json
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from newsapi import NewsApiClient
from tenacity import retry, stop_after_attempt, wait_fixed

# Setup
load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model
class TrendItem(BaseModel):
    title: str = Field(description="Concise trend title")
    summary: str = Field(description="1-sentence trend summary")
    importance: float = Field(description="Importance score 1-10")
    recency: str = Field(description="Publication time in ISO format")
    sources: List[str] = Field(description="List of source domains")

class TrendReport(BaseModel):
    topic: str
    timeframe: str
    generated_at: str
    trends: List[TrendItem]

# Tools
class NewsAPITool:
    def __init__(self):
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            raise ValueError("NEWS_API_KEY not set in .env")
        self.client = NewsApiClient(api_key=api_key)

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
    def run(self, query: str) -> str:
        from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        results = self.client.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        articles = results.get('articles', [])
        if not articles:
            return "No news results found"

        formatted = []
        for article in articles:
            source = article.get('source', {}).get('name', 'Unknown')
            published = article.get('publishedAt', 'Unknown')
            formatted.append(
                f"Source: {source}\nPublished: {published}\n"
                f"Title: {article.get('title')}\n"
                f"Description: {article.get('description')}"
            )
        return "\n\n".join(formatted)

# Combined Search Tool
class TimeAwareSearch:
    def __init__(self, frequency: str):
        logger.debug(f"Initializing TimeAwareSearch with frequency: {frequency}")
        self.time_range = self.map_frequency(frequency)
        self.newsapi_tool = NewsAPITool()
        self.ddg_tool = DuckDuckGoSearchResults(max_results=10)

    def map_frequency(self, frequency: str) -> timedelta:
        logger.debug(f"Mapping frequency '{frequency}' to time range")
        time_range = {
            "thrice_daily": timedelta(hours=8),
            "twice_daily": timedelta(hours=12),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }[frequency]
        logger.debug(f"Mapped frequency '{frequency}' to time range: {time_range}")
        return time_range

    @lru_cache(maxsize=10)
    def run(self, topic: str) -> str:
        logger.debug(f"Running search for topic: {topic}")
        time_ref = datetime.now(pytz.utc) - self.time_range
        date_filter = time_ref.strftime("%Y-%m-%d")
        logger.debug(f"Using date filter: {date_filter}")

        news_query = f"{topic} publishedAfter:{date_filter}"
        ddg_query = f"{topic} {self.get_ddg_time_keyword()}"
        logger.debug(f"NewsAPI query: {news_query}")
        logger.debug(f"DuckDuckGo query: {ddg_query}")

        news_results = self.newsapi_tool.run(news_query)
        logger.debug(f"NewsAPI results received: {len(news_results.split('\n\n'))} articles")
        
        ddg_results = self.ddg_tool.run(ddg_query)
        logger.debug(f"DuckDuckGo results received: {len(ddg_results.split('\n\n'))} results")

        return f"[NewsAPI Results]\n{news_results}\n\n[DuckDuckGo Results]\n{ddg_results}"

    def get_ddg_time_keyword(self) -> str:
        days = self.time_range.days
        hours = self.time_range.seconds // 3600
        keyword = (
            "past month" if days >= 30
            else "past week" if days >= 7
            else "past day" if days >= 1
            else "past 12 hours" if hours >= 12
            else "past 8 hours" if hours >= 8
            else "recently"
        )
        logger.debug(f"Generated DuckDuckGo time keyword: {keyword}")
        return keyword



class SearchInput(BaseModel):
    query: str = Field(description="Search topic")
    frequency: str = Field(description="Timeframe: daily, weekly, etc.")

@tool(args_schema=SearchInput)
def time_filtered_search(query: str, frequency: str) -> str:
    """Get latest news and trends about a topic from the last specified timeframe."""
    searcher = TimeAwareSearch(frequency)
    return searcher.run(query)


def generate_trend_report(topic: str, frequency: str) -> dict:
    try:
        logger.info(f"Starting trend report generation for topic: {topic}, frequency: {frequency}")
        model = ChatOpenAI(model="gpt-4", temperature=0.3)
        logger.debug("Initialized ChatOpenAI model")
        
        # Create structured output tool
        trend_report_tool = convert_to_openai_tool(TrendReport)
        logger.debug("Created trend report tool")
     
        model_with_tools = model.bind_tools(
            tools=[time_filtered_search, trend_report_tool],
            tool_choice={"type": "function", "function": {"name": "TrendReport"}}
        )
        logger.debug("Bound tools to model")

        # Create the prompt
        user_prompt = (
            f"You are a top-tier research analyst. Today is {datetime.now().strftime('%Y-%m-%d')}.\n"
            f"Generate a ranked trend report about: {topic} covering the last {frequency}.\n"
            f"Use the time_filtered_search tool to get relevant data before creating the report.\n\n"
            f"Report requirements:\n"
            f"1. Include 3-5 trends ranked by importance\n"
            f"2. Each trend must have: title, summary, importance score (1-10), "
            f"recency (ISO date), and 2-3 sources\n"
            f"3. Be concise and accurate."
        )
        logger.debug(f"Created user prompt: {user_prompt}")
        
        # Initial model call
        messages = [HumanMessage(content=user_prompt)]
        logger.debug("Making initial model call")
        response = model_with_tools.invoke(messages)
        logger.debug("Received initial model response")
        
        # Process tool calls
        tool_calls = response.additional_kwargs.get("tool_calls", [])
        logger.debug(f"Processing {len(tool_calls)} tool calls")
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            logger.debug(f"Processing tool call: {function_name} with args: {args}")
            
            if function_name == "time_filtered_search":
                logger.info("Executing time_filtered_search tool")
                search_result = time_filtered_search.invoke(args)

                logger.debug(f"Search tool returned results of length: {len(search_result)}")
                
                messages.append(ToolMessage(
                    content=search_result,
                    tool_call_id=tool_call["id"]
                ))
                logger.debug("Added search results to messages")
                
                logger.debug("Re-invoking model with search results")
                response = model_with_tools.invoke(messages)
                logger.debug("Received model response with search results")
                
            elif function_name == "TrendReport":
                logger.info("Processing TrendReport tool call")
                report = TrendReport(**args)
                logger.info(f"Generated Report: {report.json(indent=2)}")
                return {
                    "topic": report.topic,
                    "timeframe": frequency,
                    "generated_at": datetime.now().isoformat(),
                    "trends": report.trends
                }
        
        logger.error("Model failed to generate TrendReport")
        raise ValueError("Model failed to generate TrendReport")
    
    except Exception as e:
        logger.exception(f"Error generating report: {e}")
        return {
            "topic": topic,
            "timeframe": frequency,
            "generated_at": datetime.now().isoformat(),
            "trends": [],
            "error": str(e)
        }

# CLI Entry
if __name__ == "__main__":
    topic = "AI in healthcare"
    frequency = "daily"

    report = generate_trend_report(topic, frequency)
    print("=" * 60)
    print(f"Trend Report: {topic}")
    print(f"Timeframe: {report.get('timeframe')}")
    print(f"Generated at: {report.get('generated_at')}")
    print("=" * 60)

    if "error" in report:
        print(f"\n[ERROR] {report['error']}")
    elif not report.get("trends"):
        print("\nNo trends found in this timeframe.")
    else:
        for i, trend in enumerate(report["trends"], 1):
            print(f"\n{i}. {trend.title}")
            print(f"   - Summary: {trend.summary}")
            print(f"   - Importance: {trend.importance}/10")
            print(f"   - First seen: {trend.recency}")
            print(f"   - Sources: {', '.join(trend.sources)}")