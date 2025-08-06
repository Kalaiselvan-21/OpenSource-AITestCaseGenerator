from docx import Document

class WordIngestor:
    def __init__(self):
        """Initialize the Word document ingestor"""
        pass
    
    def extract_text(self, file_path):
        """
        Extract text from a Word document
        
        Args:
            file_path (str): Path to the Word document
            
        Returns:
            str: Extracted text content
        """
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    
    def save_extracted_text(self, file_path, output_file):
        """
        Extract text from a Word document and save it to a file
        
        Args:
            file_path (str): Path to the Word document
            output_file (str): Path to save the extracted text
            
        Returns:
            str: The path to the saved file
        """
        extracted_text = self.extract_text(file_path)
        
        with open(output_file, 'w') as f:
            f.write(extracted_text)
            
        return output_file


# For backward compatibility
def extract_text_from_word(file_path):
    ingestor = WordIngestor()
    return ingestor.extract_text(file_path)


# Example usage when script is run directly
if __name__ == "__main__":
    # Example: Extract text from a Word document
    word_file = 'your_word_file.docx'  # Replace with the actual file path
    
    ingestor = WordIngestor()
    extracted_text = ingestor.extract_text(word_file)
    
    # Save extracted text to a file
    output_file = 'word_doc_content.txt'
    ingestor.save_extracted_text(word_file, output_file)
    
    print(f"Extracted text from {word_file}")
