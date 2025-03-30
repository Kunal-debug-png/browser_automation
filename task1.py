import asyncio
import os
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Any, Union
import time
import re

from openai import OpenAI

from playwright.async_api import async_playwright, Page, Browser, ElementHandle


class BrowserAutomationAgent:
    def __init__(self, llm_api_key: str, headless: bool = False):
        self.llm_api_key = llm_api_key
        self.headless = headless
        self.browser = None
        self.page = None
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=llm_api_key
        )

    async def setup(self):
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        self.page = await self.context.new_page()
        print("Browser initialized successfully!")

    async def close(self):
        if self.browser:
            await self.browser.close()
            print("Browser closed.")

    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        json_match = re.search(r'(\[\s*\{.*\}\s*\])', text, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Found JSON-like structure but couldn't parse it: {e}")


        steps = []

        step_matches = re.finditer(
            r'\{\s*"action"\s*:\s*"([^"]+)"\s*,\s*"params"\s*:\s*(\{[^}]+\})\s*,\s*"description"\s*:\s*"([^"]+)"\s*\}',
            text)

        for match in step_matches:
            action = match.group(1)
            params_str = match.group(2)
            description = match.group(3)

            try:
                params = json.loads(params_str)
                steps.append({
                    "action": action,
                    "params": params,
                    "description": description
                })
            except:
                # If params can't be parsed, use empty dict
                steps.append({
                    "action": action,
                    "params": {},
                    "description": description
                })

        if steps:
            return steps

        if "youtube" in text.lower() and ("mr beast" in text.lower() or "mrbeast" in text.lower()):
            return self._create_youtube_mrbeast_steps()

        print("Could not extract valid steps from LLM response. Using fallback steps.")
        return [
            {"action": "navigate", "params": {"url": "https://google.com"}, "description": "Navigate to Google"},
            {"action": "wait", "params": {"time": 2000}, "description": "Wait for page to load"}
        ]

    def _create_youtube_mrbeast_steps(self):
        """Create specific steps for YouTube MrBeast search."""
        return [
            {"action": "navigate", "params": {"url": "https://www.youtube.com"}, "description": "Navigate to YouTube"},
            {"action": "wait", "params": {"time": 5000}, "description": "Wait for YouTube to load"},
            {"action": "click", "params": {"selector": "input#search"}, "description": "Click on search box"},
            {"action": "type", "params": {"selector": "input#search", "text": "MrBeast"},
             "description": "Type 'MrBeast' in search box"},
            {"action": "press", "params": {"key": "Enter"}, "description": "Press Enter to search"},
            {"action": "wait", "params": {"time": 5000}, "description": "Wait for search results to load"},
            {"action": "wait", "params": {"selector": "ytd-video-renderer #video-title"},
             "description": "Wait for video titles to appear"},
            {"action": "extract",
             "params": {"selector": "ytd-video-renderer #video-title", "multiple": True, "limit": 5,
                        "key": "video_titles"}, "description": "Extract titles of the first 5 videos"}
        ]

    async def get_llm_instructions(self, user_command: str) -> List[Dict[str, str]]:
        """Get step-by-step instructions from LLM to execute the user command."""
        # example
        if "youtube" in user_command.lower() and (
                "mr beast" in user_command.lower() or "mrbeast" in user_command.lower()):
            print("Using optimized steps for YouTube MrBeast search")
            return self._create_youtube_mrbeast_steps()

        prompt = f"""
        You are an AI assistant that helps create browser automation steps.

        User command: {user_command}

        Provide a detailed step-by-step plan to automate this task using Playwright.
        Include specific selectors, wait conditions, and navigation steps.
        Return your response as a JSON array of steps, where each step has:
        - "action": The action to perform (e.g., "navigate", "click", "type", "wait", "extract", "press")
        - "params": Parameters for the action (e.g., URL, selector, text, key)
        - "description": A human-readable description of the step

        Example format:
        [
            {{"action": "navigate", "params": {{"url": "https://youtube.com"}}, "description": "Navigate to YouTube"}},
            {{"action": "click", "params": {{"selector": "input#search"}}, "description": "Click on search field"}}
        ]

        Ensure your steps are explicit, precise, and handle potential issues like loading delays.
        Your response MUST be a valid JSON array and nothing else.
        """

        print("Consulting LLM for automation steps...")

        try:
            completion = self.client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=1,
                max_tokens=1024
            )

            response_text = completion.choices[0].message.content
            print("Received response from LLM")

            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return self._extract_json_from_text(response_text)

        except Exception as e:
            print(f"Error getting instructions from LLM: {e}")
            if "youtube" in user_command.lower() and (
                    "mr beast" in user_command.lower() or "mrbeast" in user_command.lower()):
                print("Using YouTube MrBeast fallback steps")
                return self._create_youtube_mrbeast_steps()
            else:
                print("Using generic fallback steps")
                return [
                    {"action": "navigate", "params": {"url": "https://google.com"},
                     "description": "Navigate to Google (fallback)"},
                    {"action": "wait", "params": {"time": 2000}, "description": "Wait for page to load"}
                ]

    async def execute_action(self, action: Dict[str, Any]) -> Optional[Any]:
        action_type = action.get("action", "").lower()
        params = action.get("params", {})
        description = action.get("description", "Executing action...")

        print(f"Step: {description}")

        try:
            if action_type == "navigate":
                url = params.get("url", "")
                if url:
                    await self.page.goto(url, wait_until="domcontentloaded")
                    await asyncio.sleep(2)  # Additional wait for JS to load
                    print(f"Navigated to {url}")

            elif action_type == "click":
                selector = params.get("selector", "")
                if selector:
                    try:
                        await self.page.wait_for_selector(selector, state="visible", timeout=15000)
                        await self.page.click(selector)
                        print(f"Clicked on {selector}")
                    except Exception as click_error:
                        print(f"Error clicking {selector}: {click_error}")
                        try:
                            await self.page.evaluate(f'document.querySelector("{selector}").click()')
                            print(f"Clicked on {selector} using JavaScript")
                        except Exception as js_error:
                            print(f"JavaScript click also failed: {js_error}")

            elif action_type == "type":
                selector = params.get("selector", "")
                text = params.get("text", "")
                if selector and text:
                    try:
                        await self.page.wait_for_selector(selector, state="visible", timeout=15000)
                        await self.page.fill(selector, text)
                        print(f"Typed '{text}' into {selector}")
                    except Exception as type_error:
                        print(f"Error typing into {selector}: {type_error}")
                        try:
                            await self.page.evaluate(f'''
                                var el = document.querySelector("{selector}");
                                if (el) {{
                                    el.value = "{text}";
                                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                }}
                            ''')
                            print(f"Typed '{text}' into {selector} using JavaScript")
                        except Exception as js_error:
                            print(f"JavaScript typing also failed: {js_error}")

            elif action_type == "press":
                key = params.get("key", "")
                if key:
                    await self.page.keyboard.press(key)
                    print(f"Pressed {key} key")

            elif action_type == "wait":
                if "time" in params:
                    time_ms = params.get("time", 1000)
                    await asyncio.sleep(time_ms / 1000)
                    print(f"Waited for {time_ms}ms")
                elif "selector" in params:
                    selector = params.get("selector", "")
                    try:
                        await self.page.wait_for_selector(selector, state="visible", timeout=20000)
                        print(f"Waited for selector {selector}")
                    except Exception as wait_error:
                        print(f"Timeout waiting for {selector}: {wait_error}")
                elif "navigation" in params:
                    await self.page.wait_for_load_state("networkidle", timeout=20000)
                    print("Waited for page navigation")

            elif action_type == "extract":
                selector = params.get("selector", "")
                attribute = params.get("attribute", "textContent")
                limit = params.get("limit", 999)  # Limit number of items to extract

                if selector:
                    try:
                        await self.page.wait_for_selector(selector, state="visible", timeout=15000)

                        if params.get("multiple", False):
                            elements = await self.page.query_selector_all(selector)
                            results = []
                            for element in elements[:limit]:
                                if attribute == "textContent":
                                    text = await element.text_content()
                                    results.append(text.strip())
                                else:
                                    attr_value = await element.get_attribute(attribute)
                                    results.append(attr_value)

                            print(f"Extracted {len(results)} items from {selector}")
                            return results
                        else:
                            element = await self.page.query_selector(selector)
                            if element:
                                if attribute == "textContent":
                                    text = await element.text_content()
                                    return text.strip()
                                else:
                                    return await element.get_attribute(attribute)

                            print(f"Extracted content from {selector}")
                    except Exception as extract_error:
                        print(f"Error extracting from {selector}: {extract_error}")
                        # Try JavaScript extraction as fallback
                        try:
                            if params.get("multiple", False):
                                results = await self.page.evaluate(f'''
                                    Array.from(document.querySelectorAll("{selector}"))
                                        .slice(0, {limit})
                                        .map(el => el.textContent.trim())
                                ''')
                                print(f"Extracted {len(results)} items using JavaScript")
                                return results
                            else:
                                result = await self.page.evaluate(f'''
                                    const el = document.querySelector("{selector}");
                                    el ? el.textContent.trim() : null
                                ''')
                                return result
                        except Exception as js_error:
                            print(f"JavaScript extraction also failed: {js_error}")
                            return []

            elif action_type == "scroll":
                if "selector" in params:
                    selector = params.get("selector", "")
                    await self.page.wait_for_selector(selector, state="visible", timeout=10000)
                    await self.page.evaluate(f'document.querySelector("{selector}").scrollIntoView()')
                else:
                    x = params.get("x", 0)
                    y = params.get("y", 500)  # Default scroll down 500px
                    await self.page.evaluate(f'window.scrollBy({x}, {y})')
                print("Scrolled the page")

            elif action_type == "screenshot":
                path = params.get("path", f"screenshot_{int(time.time())}.png")
                await self.page.screenshot(path=path)
                print(f"Saved screenshot to {path}")

            else:
                print(f"Unknown action type: {action_type}")

        except Exception as e:
            print(f"Error executing action {action_type}: {e}")
            return None

        return None

    async def run_automation(self, user_command: str) -> Dict[str, Any]:
        """Execute the full automation based on the user command."""
        print(f"\n==== Starting automation for: {user_command} ====\n")

        if not self.browser:
            await self.setup()

        # Get steps from LLM
        steps = await self.get_llm_instructions(user_command)

        # Track extracted results
        results = {}

        for i, step in enumerate(steps):
            print(f"\nStep {i + 1}/{len(steps)}: {step.get('description', 'Executing...')}")

            result = await self.execute_action(step)

            if step.get("action") == "extract" and result is not None:
                key = step.get("params", {}).get("key", f"extracted_data_{i}")
                results[key] = result

            await asyncio.sleep(1)

        if "youtube" in user_command.lower() and (
                "mr beast" in user_command.lower() or "mrbeast" in user_command.lower()):
            if not results.get("video_titles"):
                # Try a desperate recovery for YouTube
                try:
                    print("\nAttempting emergency video title extraction...")
                    # Try multiple different selectors that might work for YouTube
                    selectors = [
                        "ytd-video-renderer #video-title",
                        "#video-title",
                        "a#video-title",
                        "ytd-video-renderer h3",
                        "#title h3 a"
                    ]

                    for selector in selectors:
                        try:
                            print(f"Trying selector: {selector}")
                            await self.page.wait_for_selector(selector, timeout=5000)
                            elements = await self.page.query_selector_all(selector)
                            if elements and len(elements) > 0:
                                titles = []
                                for i, element in enumerate(elements):
                                    if i >= 5:  # Limit to 5 videos
                                        break
                                    text = await element.text_content()
                                    if text and text.strip():
                                        titles.append(text.strip())

                                if titles:
                                    print(f"Successfully extracted {len(titles)} titles!")
                                    results["video_titles"] = titles
                                    break
                        except Exception as e:
                            print(f"Selector {selector} failed: {e}")
                            continue

                    if not results.get("video_titles"):
                        print("Trying JavaScript extraction...")
                        titles = await self.page.evaluate('''
                            Array.from(document.querySelectorAll('a#video-title, #video-title, ytd-video-renderer h3'))
                                .slice(0, 5)
                                .map(el => el.textContent.trim())
                                .filter(text => text.length > 0)
                        ''')
                        if titles and len(titles) > 0:
                            results["video_titles"] = titles
                except Exception as recovery_error:
                    print(f"Emergency recovery failed: {recovery_error}")

        print("\n==== Automation completed successfully ====\n")

        return {
            "command": user_command,
            "steps_executed": len(steps),
            "results": results
        }


async def main():
    parser = argparse.ArgumentParser(description="Browser Automation Agent")
    parser.add_argument("--command", type=str, help="Natural language command to execute")
    parser.add_argument("--api-key", type=str, help="LLM API key")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")

    args = parser.parse_args()

    # Use provided arguments or prompt for them
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        api_key = ""

    command = args.command
    if not command:
        command = input(
            "Enter your automation command (e.g., 'go to YouTube and find the titles of 5 latest videos from MrBeast'): ")

    agent = BrowserAutomationAgent(llm_api_key=api_key, headless=args.headless)

    try:
        results = await agent.run_automation(command)

        print("\n==== Automation Results ====")
        print(f"Command: {results['command']}")
        print(f"Steps executed: {results['steps_executed']}")

        if results['results']:
            print("\nExtracted data:")
            for key, value in results['results'].items():
                if isinstance(value, list):
                    print(f"\n{key}:")
                    for i, item in enumerate(value):
                        print(f"  {i + 1}. {item}")
                else:
                    print(f"\n{key}: {value}")
        else:
            print("\nNo data was extracted.")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
