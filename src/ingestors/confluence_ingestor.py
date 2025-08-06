from atlassian import Confluence

class ConfluenceIngestor:
    def __init__(self, confluence_url=None, username=None, api_token=None):
        """
        Initialize the Confluence ingestor
        
        Args:
            confluence_url (str): URL of the Confluence instance
            username (str): Username for authentication
            api_token (str): API token for authentication
        """
        self.confluence_url = confluence_url
        self.username = username
        self.api_token = api_token
        self.confluence = None
        
        if confluence_url and username and api_token:
            self.connect()
    
    def connect(self, confluence_url=None, username=None, api_token=None):
        """
        Connect to Confluence API
        
        Args:
            confluence_url (str, optional): URL of the Confluence instance
            username (str, optional): Username for authentication
            api_token (str, optional): API token for authentication
        """
        # Use provided parameters or fall back to instance variables
        url = confluence_url or self.confluence_url
        user = username or self.username
        token = api_token or self.api_token
        
        if not (url and user and token):
            raise ValueError("Confluence URL, username, and API token are required")
        
        # Initialize the Confluence API client
        self.confluence = Confluence(
            url=url,
            username=user,
            password=token
        )
        
        # Update instance variables
        self.confluence_url = url
        self.username = user
        self.api_token = token
    
    def fetch_page_by_id(self, page_id):
        """
        Fetch a Confluence page by its ID
        
        Args:
            page_id (str): The ID of the Confluence page
            
        Returns:
            str: The HTML content of the page
        """
        if not self.confluence:
            raise RuntimeError("Not connected to Confluence. Call connect() first.")
            
        content = self.confluence.get_page_by_id(page_id, expand='body.storage')
        page_content = content['body']['storage']['value']
        return page_content
    
    def save_page_content(self, page_id, output_file):
        """
        Fetch a page and save its content to a file
        
        Args:
            page_id (str): The ID of the Confluence page
            output_file (str): Path to save the content
            
        Returns:
            str: The path to the saved file
        """
        page_content = self.fetch_page_by_id(page_id)
        
        with open(output_file, 'w') as f:
            f.write(page_content)
            
        return output_file


# For backward compatibility
def fetch_confluence_page(page_id, confluence_url='https://yourcompany.atlassian.net/wiki', 
                         username='your_email@domain.com', api_token='your_api_token'):
    ingestor = ConfluenceIngestor(confluence_url, username, api_token)
    return ingestor.fetch_page_by_id(page_id)


# Example usage when script is run directly
if __name__ == "__main__":
    # Set your Confluence credentials here
    confluence_url = 'https://yourcompany.atlassian.net/wiki'
    username = 'your_email@domain.com'
    api_token = 'your_api_token'
    
    # Example: Fetch a page by its ID
    page_id = 'your_page_id_here'  # Replace with an actual Confluence page ID
    
    ingestor = ConfluenceIngestor(confluence_url, username, api_token)
    page_content = ingestor.fetch_page_by_id(page_id)
    
    # Save content to a file
    output_file = 'confluence_page_content.html'
    ingestor.save_page_content(page_id, output_file)
    
    print(f"Fetched content from Confluence page {page_id}")
