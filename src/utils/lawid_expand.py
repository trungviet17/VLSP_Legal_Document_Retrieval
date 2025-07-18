import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json 
import requests 
from playwright.sync_api import sync_playwright
from tqdm import tqdm
import time 
import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class ExpandContext : 

    def __init__(self, file_path: str): 

        self.data =  self._load_data(file_path)

    def _load_data(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data 


    def get_context_with_vbpl(self): 

        base_url = "https://vbpl.vn/pages/vbpq-timkiem.aspx?type=0&s=1&SearchIn=Title,Title1&Keyword="
        
        with sync_playwright() as p: 

            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            )

            page = context.new_page()

            try: 
                for item in tqdm(self.data, desc="Expanding context", unit="item"):
                    keyword = item.get("law_id", "")
                    url = base_url + keyword

                    if "law_context" in item:
                        continue
                    # logger.info(f"Processing URL: {url}")
                    max_retries = 3
                    found = False
                    for _ in range(max_retries):
                        try:
                            page.goto(url, timeout=10000)
                            page.wait_for_selector("ul.listLaw", timeout=10000)

                            content_items = page.query_selector_all("ul.listLaw div.item")

                            if not content_items: 
                                # logger.warning(f"No content items found for keyword: {keyword}")
                                continue 

                            for content_item in content_items:
                                title = content_item.query_selector("p.title")
                                if not title: 
                                    # logger.warning(f"No title found for content item with keyword: {keyword}")
                                    continue 

                                title_text = title.text_content().strip()

                                if keyword in title_text:
                                    des_element = content_item.query_selector("div.des")
                                    if des_element: 
                                        item["law_context"] = des_element.text_content().strip()
                                        # logger.info(f"Context found for {item.get('title', 'Unknown')}: {item['law_context']}")
                                        found = True
                                        break 

                        except Exception as e:
                           print(f"An error occurred while processing {url}: {e}")

                        if found:
                            break
                    if not found:
                       item["law_context"] = "Context not found"
                       # logger.warning(f"Context not found for: {item.get('title', 'Unknown')}")

                    if self.data.index(item) % 10 == 0:
                        self.save_context("../data/vlsp/legal_corpus_with_context.json")

            except Exception as e:
                logger.error(f"An error occurred: {e}")
            finally:
                browser.close()
       

        print("Context expansion completed successfully.")
        self.save_context("../data/vlsp/legal_corpus_with_context.json")
        return self.data


    def get_content_with_tvpl(self): 
        pass 








    def save_context(self, output_path: str):

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)
        logger.info(f"Context data saved to {output_path}")


if __name__ == "__main__":
    file_path = "../data/vlsp/legal_corpus_with_context.json"
    context_expander = ExpandContext(file_path)
    context_data = context_expander.get_context()
    print("Context expansion completed.") 


    


    