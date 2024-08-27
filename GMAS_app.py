import asyncio
import json
from typing import List, Dict
from duckduckgo_search import DDGS
import aiohttp
import os

# Note: Replace these with your actual API keys and endpoints
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

class Agent:
    def __init__(self, name: str):
        self.name = name

    async def process_task(self, task: Dict) -> Dict:
        raise NotImplementedError

class ResearchAgent(Agent):
    async def process_task(self, task: Dict) -> Dict:
        query = task['query']
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        return {'research_results': results}

class WritingAgent(Agent):
    async def process_task(self, task: Dict) -> Dict:
        prompt = f"Write a blog post about {task['topic']} using the following research: {task['research_results']}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/chat/completions?api-version=2023-05-15",
                headers={"api-key": AZURE_OPENAI_API_KEY},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000
                }
            ) as response:
                result = await response.json()
                blog_post = result['choices'][0]['message']['content']
        
        return {'blog_post': blog_post}

class EditingAgent(Agent):
    async def process_task(self, task: Dict) -> Dict:
        prompt = f"Edit and improve the following blog post: {task['blog_post']}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/chat/completions",
                headers={"x-api-key": ANTHROPIC_API_KEY},
                json={
                    "model": "claude-3-opus-20240229",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000
                }
            ) as response:
                result = await response.json()
                edited_post = result['content'][0]['text']
        
        return {'edited_post': edited_post}

class SEOAgent(Agent):
    async def process_task(self, task: Dict) -> Dict:
        prompt = f"Generate SEO metadata for the following blog post: {task['edited_post']}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/chat/completions?api-version=2023-05-15",
                headers={"api-key": AZURE_OPENAI_API_KEY},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                }
            ) as response:
                result = await response.json()
                seo_metadata = result['choices'][0]['message']['content']
        
        return {'seo_metadata': seo_metadata}

class CentralGuidingAgent:
    def __init__(self):
        self.agents = {
            'research': ResearchAgent('Research'),
            'writing': WritingAgent('Writing'),
            'editing': EditingAgent('Editing'),
            'seo': SEOAgent('SEO')
        }
        self.task_queue = asyncio.Queue()
        self.results = {}

    async def add_task(self, task: Dict):
        await self.task_queue.put(task)

    async def process_tasks(self):
        while not self.task_queue.empty():
            task = await self.task_queue.get()
            agent = self.agents[task['agent']]
            result = await agent.process_task(task)
            self.results.update(result)
            self.task_queue.task_done()

    async def run_pipeline(self, topic: str):
        await self.add_task({'agent': 'research', 'query': topic})
        await self.process_tasks()
        
        await self.add_task({'agent': 'writing', 'topic': topic, 'research_results': self.results['research_results']})
        await self.process_tasks()
        
        await self.add_task({'agent': 'editing', 'blog_post': self.results['blog_post']})
        await self.process_tasks()
        
        await self.add_task({'agent': 'seo', 'edited_post': self.results['edited_post']})
        await self.process_tasks()
        
        return {
            'blog_post': self.results['edited_post'],
            'seo_metadata': self.results['seo_metadata']
        }

async def main():
    central_agent = CentralGuidingAgent()
    result = await central_agent.run_pipeline("The Impact of Artificial Intelligence on Modern HR Operations")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
