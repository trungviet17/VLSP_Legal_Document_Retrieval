import os, sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json 
import requests 
from bs4 import BeautifulSoup
from tqdm import tqdm
import time 


class ExpandContext : 

    def __init__(self, file_path: str): 
        self.data =  self._load_data(file_path)

    def _load_data(self, file_path):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data 


    def get_context(self): 

        base_url = "https://vbpl.vn/pages/vbpq-timkiem.aspx?type=0&s=1&SearchIn=Title,Title1&Keyword="
        headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

        for item in tqdm(self.data, desc="Processing items"):
            keyword = item.get('law_id', '')
            url = base_url + keyword
            print(url)
            try: 
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  

                soup = BeautifulSoup(response.text, 'html.parser')
                time.sleep(2)
                
                content = soup.find("ul", class_="listLaw")
                if not content: 
                    item["law_context"] = "No content found"
                    print(f"No content found for {keyword}")
                    continue 

                content_items = content.find_all("div", class_= "item")
                
                for content_item in content_items: 
                    title = content_item.find("p", class_="title").get_text(strip=True)
                    if keyword in title: 
                        item["law_context"] = content_item.find("div", class_ = "des").get_text(strip=True)
                        break
                self.save_context("../data/vlsp/legal_corpus_with_context.json")
                        
            except requests.RequestException as e:
                print(f"Error fetching data for {keyword}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred for {keyword}: {e}")


        return self.data

    def save_context(self, output_path: str):

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)
        print(f"Context saved to {output_path}")

if __name__ == "__main__":
    file_path = "../data/vlsp/legal_corpus.json"
    context_expander = ExpandContext(file_path)
    context_data = context_expander.get_context()
    print("Context expansion completed.") 


    


    