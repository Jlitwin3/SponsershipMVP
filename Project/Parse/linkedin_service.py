import os
import requests
import json
from datetime import datetime

class LinkedInService:
    def __init__(self, api_key=None, provider="rapidapi"):
        """
        Initialize the LinkedIn Service.
        
        Args:
            api_key (str): API Key for the chosen provider. Defaults to env var LINKEDIN_API_KEY (or RAPIDAPI_KEY).
            provider (str): The API provider to use. Currently supports 'proxycurl' or 'rapidapi' (default).
        """
        self.provider = provider
        
        if provider == "rapidapi":
            self.api_key = api_key or os.getenv("RAPIDAPI_KEY")
            self.host = os.getenv("RAPIDAPI_HOST", "fresh-linkedin-profile-data.p.rapidapi.com")
        else:
            self.api_key = api_key or os.getenv("RAPIDAPI_KEY")

        if not self.api_key:
            print(f"⚠️  Warning: API Key not found for {provider}. LinkedIn service will return mock data.")

    def get_company_updates(self, company_linkedin_url, limit=5):
        """
        Fetch recent updates/posts from a company's LinkedIn page.
        
        Args:
            company_linkedin_url (str): The full LinkedIn URL of the company (e.g., https://www.linkedin.com/company/university-of-oregon/)
            limit (int): Number of posts to retrieve.
            
        Returns:
            list: A list of dictionaries containing post content and metadata.
        """
        if not self.api_key:
            return self._get_mock_data(company_linkedin_url)

        if self.provider == "proxycurl":
            return self._fetch_proxycurl(company_linkedin_url, limit)
        elif self.provider == "rapidapi":
            return self._fetch_rapidapi(company_linkedin_url, limit)
        else:
            print(f"❌ Provider '{self.provider}' not implemented yet.")
            return []

    def _fetch_rapidapi(self, company_url, limit):
        """
        Fetch data using Fresh LinkedIn Scraper via RapidAPI.
        """
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host
        }

        # Step 1: Resolve Company URL to ID
        # Endpoint: /get-company-by-linkedinurl
        resolve_url = f"https://{self.host}/get-company-by-linkedinurl"
        querystring = {"linkedin_url": company_url}
        
        company_id = None
        company_name = "Unknown"
        
        try:
            response = requests.get(resolve_url, headers=headers, params=querystring)
            if response.status_code == 200:
                data = response.json()
                # print(f"🐛 Raw API Response: {json.dumps(data, indent=2)}")
                
                # Try to find ID in response
                if data and 'data' in data and data['data']:
                     # API returns 'company_id' inside 'data'
                     company_id = data['data'].get('company_id') or data['data'].get('id')
                     company_name = data['data'].get('company_name', 'Unknown')
                elif 'id' in data:
                     company_id = data['id']
                
                print(f"✅ Resolved Company ID: {company_id} ({company_name})")
            else:
                print(f"❌ RapidAPI Resolve Error: {response.status_code} - {response.text}")
                # Fallback: Try to extract from URL if it's a numeric URL, otherwise fail
                # But usually URL is /company/name
                return []
        except Exception as e:
            print(f"❌ RapidAPI Resolve Exception: {e}")
            return []

        if not company_id:
            print("❌ Could not resolve Company ID from URL.")
            return []

        # Step 2: Get Company Posts
        # Endpoint: /get-company-posts
        posts_url = f"https://{self.host}/get-company-posts"
        params = {
            "linkedin_url": company_url,
            "start": "0",
            "count": str(limit)
        }
        
        try:
            print(f"📥 Fetching posts for Company ID: {company_id}...")
            response = requests.get(posts_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return self._format_rapidapi_response(data, company_name)
            else:
                print(f"❌ RapidAPI Posts Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"❌ RapidAPI Posts Exception: {e}")
            return []

    def _format_rapidapi_response(self, data, company_name):
        """
        Format RapidAPI response.
        Structure depends on the specific API, but usually 'data' is a list of posts.
        """
        chunks = []
        posts = data.get('data', [])
        
        if not posts and isinstance(data, list):
            posts = data
            
        for post in posts:
            # Extract text
            text = post.get('text', '') or post.get('summary', '')
            if not text:
                continue
                
            # Extract date
            date_str = post.get('posted_date', '') or datetime.now().isoformat()
            
            # Extract link
            post_url = post.get('url', '') or post.get('share_url', '')
            
            content = f"Company: {company_name}\nPost Date: {date_str}\n\n{text}"
            if post_url:
                content += f"\n\nLink: {post_url}"
                
            chunks.append({
                "content": content,
                "source": "LinkedIn Post (RapidAPI)",
                "date": date_str,
                "url": post_url
            })
            
        return chunks

    def _fetch_proxycurl(self, company_url, limit):
        """
        Fetch data using Proxycurl API.
        Docs: https://nubela.co/proxycurl/docs
        """
        # Note: This is a simplified implementation. 
        # Proxycurl doesn't have a direct "company posts" endpoint in the same way as scraping,
        # but they have a 'Company Profile Endpoint' that returns some data.
        # For posts specifically, we might need a different provider or a specific scraping API.
        # 
        # However, for this MVP, let's assume we are using a service that CAN get posts 
        # or we are getting the company profile description and specialties.
        
        # Let's try to get the Company Profile first as it's the most standard request.
        api_endpoint = 'https://rapidapi.com/rockapis-rockapis-default/api/linkedin-data-api'
        headers = {'Authorization': 'Bearer ' + self.api_key}
        params = {
            'url': company_url,
            'resolve_numeric_id': 'true',
            'categories': 'include',
            'funding_data': 'include',
            'extra': 'include',
            'exit_data': 'include',
            'acquisitions': 'include',
            'use_cache': 'if-present',
        }
        
        try:
            response = requests.get(api_endpoint, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return self._format_proxycurl_response(data)
            else:
                print(f"❌ Proxycurl Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"❌ LinkedIn Service Error: {e}")
            return []

    def _format_proxycurl_response(self, data):
        """
        Format the raw API response into a standard list of text chunks.
        """
        chunks = []
        
        # 1. Basic Info
        name = data.get('name', 'Unknown Company')
        description = data.get('description', '')
        website = data.get('website', '')
        
        if description:
            chunks.append({
                "content": f"Company: {name}\nWebsite: {website}\n\nDescription:\n{description}",
                "source": "LinkedIn Profile",
                "date": datetime.now().isoformat()
            })
            
        # 2. Specialties
        specialties = data.get('specialties', [])
        if specialties:
            chunks.append({
                "content": f"Company: {name}\nSpecialties:\n" + ", ".join(specialties),
                "source": "LinkedIn Specialties",
                "date": datetime.now().isoformat()
            })
            
        # 3. Recent Updates (if available in the response - Proxycurl usually separates this)
        # For now, we'll return what we have.
        
        return chunks

    def _get_mock_data(self, company_url):
        """
        Return mock data for testing without an API key.
        """
        print(f"🧪 Returning MOCK LinkedIn data for {company_url}")
        return [
            {
                "content": f"MOCK DATA: {company_url} recently announced a new partnership with Nike to support student athletes.",
                "source": "LinkedIn Mock",
                "date": datetime.now().isoformat()
            },
            {
                "content": f"MOCK DATA: The University of Oregon is launching a new sustainability initiative sponsored by Columbia Sportswear.",
                "source": "LinkedIn Mock",
                "date": datetime.now().isoformat()
            }
        ]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test the service
    service = LinkedInService()
    updates = service.get_company_updates("https://www.linkedin.com/school/university-of-oregon/")
    for update in updates:
        print("---")
        print(update['content'])
