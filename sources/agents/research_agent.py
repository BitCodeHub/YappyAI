"""
Research Agent - Autonomous research and information gathering
"""
import json
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import requests

from sources.agents.agent import Agent
from sources.logger import Logger
from sources.tools.webSearch import WebSearch
from sources.browser import SimpleBrowser


class ResearchAgent(Agent):
    """
    Specialized agent for conducting autonomous research
    """
    
    def __init__(self, llm_provider=None):
        super().__init__(llm_provider)
        self.logger = Logger("research_agent.log")
        self.web_search = WebSearch()
        self.browser = SimpleBrowser()
        self.research_cache = {}
        
    def generate_response(self, query: str) -> Tuple[str, str]:
        """
        Main entry point for research requests
        """
        self.logger.log(f"Research query: {query}")
        
        # Determine research type and depth
        research_plan = self._create_research_plan(query)
        
        # Execute research
        research_results = self._execute_research(research_plan)
        
        # Synthesize findings
        synthesis = self._synthesize_research(research_results, query)
        
        # Generate report
        report = self._generate_report(synthesis, research_plan)
        
        return report, "research"
        
    def _create_research_plan(self, query: str) -> Dict:
        """Create a structured research plan"""
        # Use LLM to analyze query and create plan
        system_prompt = """You are a research planning assistant. 
        Analyze the query and create a structured research plan."""
        
        user_prompt = f"""
        Create a research plan for: {query}
        
        Include:
        1. Main research questions
        2. Key topics to investigate
        3. Types of sources needed
        4. Research depth (quick/standard/comprehensive)
        5. Any specific constraints or requirements
        
        Return as JSON.
        """
        
        # Simplified plan for now
        plan = {
            "query": query,
            "main_questions": self._extract_questions(query),
            "topics": self._extract_topics(query),
            "source_types": ["web", "academic", "news"],
            "depth": "standard",
            "max_sources": 10,
            "time_limit": 300  # 5 minutes
        }
        
        self.logger.log(f"Created research plan: {json.dumps(plan, indent=2)}")
        return plan
        
    def _extract_questions(self, query: str) -> List[str]:
        """Extract research questions from query"""
        questions = []
        
        # Direct questions
        if "?" in query:
            questions.append(query)
        else:
            # Convert statement to questions
            if "research" in query.lower():
                topic = query.replace("research", "").replace("Research", "").strip()
                questions.extend([
                    f"What is {topic}?",
                    f"What are the key aspects of {topic}?",
                    f"What are the latest developments in {topic}?"
                ])
            elif "compare" in query.lower():
                questions.append(f"What are the key differences?")
                questions.append(f"What are the similarities?")
            else:
                questions.append(f"What information is available about {query}?")
                
        return questions
        
    def _extract_topics(self, query: str) -> List[str]:
        """Extract key topics from query"""
        # Simple keyword extraction
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        words = query.lower().split()
        topics = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Add the full query as a topic too
        topics.insert(0, query)
        
        return topics[:5]  # Limit to 5 topics
        
    def _execute_research(self, plan: Dict) -> Dict:
        """Execute the research plan"""
        results = {
            "sources": [],
            "raw_data": [],
            "timestamps": {},
            "errors": []
        }
        
        start_time = datetime.now()
        
        # Search for each topic
        for topic in plan["topics"][:3]:  # Limit topics for performance
            try:
                # Web search
                search_results = self._search_web(topic, plan["max_sources"] // len(plan["topics"]))
                
                # Process each result
                for result in search_results:
                    source_data = self._process_source(result)
                    if source_data:
                        results["sources"].append(source_data)
                        results["raw_data"].append({
                            "url": result.get("url"),
                            "title": result.get("title"),
                            "content": source_data.get("content", ""),
                            "timestamp": datetime.now().isoformat()
                        })
                        
            except Exception as e:
                self.logger.error(f"Error researching topic '{topic}': {e}")
                results["errors"].append(f"Failed to research {topic}: {str(e)}")
                
            # Check time limit
            if (datetime.now() - start_time).seconds > plan["time_limit"]:
                self.logger.log("Research time limit reached")
                break
                
        results["timestamps"]["completed"] = datetime.now().isoformat()
        return results
        
    def _search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform web search"""
        try:
            # Use the web search tool
            search_response = self.web_search.execute(query)
            
            # Parse results
            results = []
            if isinstance(search_response, str):
                # Extract URLs and titles from response
                lines = search_response.split('\n')
                for line in lines[:max_results]:
                    if 'http' in line:
                        # Simple extraction
                        parts = line.split(' - ')
                        if len(parts) >= 2:
                            results.append({
                                "title": parts[0].strip(),
                                "url": parts[1].strip() if 'http' in parts[1] else "",
                                "snippet": parts[2] if len(parts) > 2 else ""
                            })
                            
            return results
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []
            
    def _process_source(self, source: Dict) -> Optional[Dict]:
        """Process a single source"""
        try:
            url = source.get("url", "")
            if not url or not url.startswith("http"):
                return None
                
            # Try to fetch content
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Research Agent)'
            })
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return {
                    "url": url,
                    "title": source.get("title", soup.title.string if soup.title else ""),
                    "content": text[:5000],  # Limit content length
                    "source_type": "web",
                    "credibility": self._assess_credibility(url)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to process source {source.get('url')}: {e}")
            return None
            
    def _assess_credibility(self, url: str) -> float:
        """Assess source credibility (0-1)"""
        credibility = 0.5  # Default
        
        # Known credible domains
        credible_domains = [
            ".edu", ".gov", ".org",
            "wikipedia.org", "reuters.com", "apnews.com",
            "nature.com", "sciencedirect.com", "arxiv.org"
        ]
        
        for domain in credible_domains:
            if domain in url:
                credibility += 0.3
                break
                
        # Penalize certain domains
        low_cred_domains = ["blog", "forum", "reddit.com"]
        for domain in low_cred_domains:
            if domain in url:
                credibility -= 0.2
                break
                
        return max(0, min(1, credibility))
        
    def _synthesize_research(self, results: Dict, original_query: str) -> Dict:
        """Synthesize research findings"""
        synthesis = {
            "summary": "",
            "key_findings": [],
            "sources_used": len(results["sources"]),
            "confidence": 0.0,
            "gaps": []
        }
        
        if not results["sources"]:
            synthesis["summary"] = "No reliable sources found for this query."
            synthesis["gaps"] = ["Unable to find relevant information"]
            return synthesis
            
        # Analyze content
        all_content = " ".join(s["content"] for s in results["sources"])
        
        # Extract key points (simplified)
        key_points = []
        for source in results["sources"]:
            if source["credibility"] > 0.6:
                # Extract first few sentences as key points
                sentences = source["content"].split('. ')[:3]
                key_points.extend(sentences)
                
        synthesis["key_findings"] = list(set(key_points))[:5]
        
        # Create summary
        synthesis["summary"] = self._create_summary(results["sources"], original_query)
        
        # Calculate confidence
        avg_credibility = sum(s["credibility"] for s in results["sources"]) / len(results["sources"])
        synthesis["confidence"] = avg_credibility
        
        return synthesis
        
    def _create_summary(self, sources: List[Dict], query: str) -> str:
        """Create a summary from sources"""
        # In production, this would use LLM
        # For now, create simple summary
        
        summary_parts = [
            f"Based on {len(sources)} sources, here's what I found about '{query}':",
            ""
        ]
        
        # Group by credibility
        high_cred = [s for s in sources if s["credibility"] > 0.7]
        med_cred = [s for s in sources if 0.4 <= s["credibility"] <= 0.7]
        
        if high_cred:
            summary_parts.append("From highly credible sources:")
            for source in high_cred[:3]:
                snippet = source["content"][:200] + "..."
                summary_parts.append(f"• {snippet}")
                
        if med_cred:
            summary_parts.append("\nFrom additional sources:")
            for source in med_cred[:2]:
                snippet = source["content"][:200] + "..."
                summary_parts.append(f"• {snippet}")
                
        return "\n".join(summary_parts)
        
    def _generate_report(self, synthesis: Dict, plan: Dict) -> str:
        """Generate final research report"""
        report_parts = [
            "# Research Report",
            f"**Query**: {plan['query']}",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Sources Analyzed**: {synthesis['sources_used']}",
            f"**Confidence Level**: {synthesis['confidence']:.1%}",
            "",
            "## Summary",
            synthesis["summary"],
            "",
            "## Key Findings"
        ]
        
        for i, finding in enumerate(synthesis["key_findings"], 1):
            report_parts.append(f"{i}. {finding}")
            
        if synthesis["gaps"]:
            report_parts.extend([
                "",
                "## Information Gaps",
                "The following areas need more research:"
            ])
            for gap in synthesis["gaps"]:
                report_parts.append(f"- {gap}")
                
        report_parts.extend([
            "",
            "---",
            "*This report was generated by the Research Agent*"
        ])
        
        return "\n".join(report_parts)
        
    async def autonomous_research(self, topic: str, depth: str = "standard", 
                                 output_format: str = "report") -> Dict:
        """
        Conduct autonomous research on a topic
        
        Args:
            topic: Research topic
            depth: Research depth (quick, standard, comprehensive)
            output_format: Output format (summary, report, data)
            
        Returns:
            Research results dictionary
        """
        self.logger.log(f"Starting autonomous research on: {topic}")
        
        # Create research plan
        plan = {
            "query": topic,
            "depth": depth,
            "output_format": output_format,
            "max_sources": {"quick": 5, "standard": 10, "comprehensive": 20}[depth],
            "time_limit": {"quick": 120, "standard": 300, "comprehensive": 600}[depth]
        }
        
        # Execute research
        results = self._execute_research(plan)
        
        # Synthesize
        synthesis = self._synthesize_research(results, topic)
        
        # Format output
        if output_format == "summary":
            output = synthesis["summary"]
        elif output_format == "report":
            output = self._generate_report(synthesis, plan)
        else:  # data
            output = {
                "synthesis": synthesis,
                "sources": results["sources"],
                "metadata": {
                    "topic": topic,
                    "depth": depth,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        return {
            "success": True,
            "output": output,
            "sources_count": len(results["sources"]),
            "confidence": synthesis["confidence"]
        }