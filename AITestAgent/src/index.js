import Resolver from '@forge/resolver';
import api, { route } from '@forge/api';

const resolver = new Resolver();

// Backend API URL
const BACKEND_URL = 'http://localhost:5002';

// Helper function to make API calls to the backend
async function callBackend(endpoint, method = 'GET', body = null) {
  const url = `${BACKEND_URL}/${endpoint}`;
  
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json'
    }
  };
  
  if (body) {
    options.body = JSON.stringify(body);
  }
  
  try {
    const response = await api.fetch(url, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Error calling backend: ${response.status} ${response.statusText}`, errorText);
      return { 
        success: false, 
        error: `Backend error: ${response.status} ${response.statusText}` 
      };
    }
    
    const data = await response.json();
    return { success: true, ...data };
  } catch (error) {
    console.error('Error calling backend:', error);
    return { 
      success: false, 
      error: `Failed to connect to backend: ${error.message}` 
    };
  }
}

// Main function for the issue panel
resolver.define('run', async ({ context }) => {
  console.log('AI Test Case Generator running');
  return { success: true };
});

// Function to get issue details
resolver.define('getIssueDetails', async ({ payload, context }) => {
  const { issueKey } = payload;
  
  if (!issueKey) {
    return { success: false, error: 'Issue key is required' };
  }
  
  try {
    // Get issue details from Jira API
    const issueResponse = await api.asUser().requestJira(route`/rest/api/3/issue/${issueKey}`);
    
    if (!issueResponse.ok) {
      const errorText = await issueResponse.text();
      console.error(`Error fetching issue: ${issueResponse.status}`, errorText);
      return { success: false, error: `Failed to fetch issue: ${issueResponse.status}` };
    }
    
    const issueData = await issueResponse.json();
    
    // Extract relevant fields
    const summary = issueData.fields.summary;
    const description = issueData.fields.description ? 
      extractTextFromADF(issueData.fields.description) : '';
    
    // Try to get acceptance criteria from custom field
    let acceptanceCriteria = '';
    
    // Look for common acceptance criteria field names
    const possibleFieldNames = [
      'customfield_10016', // Common for acceptance criteria
      'customfield_10017',
      'customfield_10018',
      'customfield_10019',
      'customfield_10020'
    ];
    
    for (const fieldName of possibleFieldNames) {
      if (issueData.fields[fieldName]) {
        const fieldValue = issueData.fields[fieldName];
        if (typeof fieldValue === 'object' && fieldValue !== null) {
          // Handle Atlassian Document Format
          acceptanceCriteria = extractTextFromADF(fieldValue);
        } else if (typeof fieldValue === 'string') {
          acceptanceCriteria = fieldValue;
        }
        
        if (acceptanceCriteria) {
          break;
        }
      }
    }
    
    return {
      success: true,
      data: {
        key: issueKey,
        summary,
        description,
        acceptanceCriteria
      }
    };
  } catch (error) {
    console.error('Error getting issue details:', error);
    return { success: false, error: `Failed to get issue details: ${error.message}` };
  }
});

// Function to generate test cases directly
resolver.define('generateTestCasesDirectly', async ({ payload }) => {
  try {
    // Call the backend API to generate test cases
    const result = await callBackend('generate-test-cases', 'POST', payload);
    return result;
  } catch (error) {
    console.error('Error generating test cases:', error);
    return { success: false, error: `Failed to generate test cases: ${error.message}` };
  }
});

// Function to export test cases
resolver.define('exportTestCases', async ({ payload }) => {
  const { testCases, format } = payload;
  
  if (!testCases) {
    return { success: false, error: 'Test cases are required' };
  }
  
  try {
    // For now, just return the test cases in the requested format
    // In the future, this could call a backend endpoint to format the test cases
    let formattedTestCases = testCases;
    
    if (format === 'jira') {
      formattedTestCases = convertToJiraFormat(testCases);
    } else if (format === 'html') {
      formattedTestCases = convertToHtmlFormat(testCases);
    }
    // Default is markdown, which is already the format we use
    
    return { success: true, formattedTestCases };
  } catch (error) {
    console.error('Error exporting test cases:', error);
    return { success: false, error: `Failed to export test cases: ${error.message}` };
  }
});

// Helper function to extract text from Atlassian Document Format
function extractTextFromADF(adf) {
  if (!adf || !adf.content) {
    return '';
  }
  
  let text = '';
  
  function processNode(node) {
    if (node.text) {
      text += node.text;
    }
    
    if (node.content) {
      for (const child of node.content) {
        processNode(child);
        
        // Add newlines for paragraph and heading nodes
        if (child.type === 'paragraph' || child.type?.startsWith('heading')) {
          text += '\n';
        }
        
        // Add bullet points for list items
        if (child.type === 'listItem') {
          text += '- ';
        }
      }
    }
  }
  
  processNode(adf);
  return text.trim();
}

// Helper function to convert test cases to Jira format
function convertToJiraFormat(testCases) {
  // Simple conversion to Jira markup
  return testCases
    .replace(/^# (.*)/gm, 'h1. $1')
    .replace(/^## (.*)/gm, 'h2. $1')
    .replace(/^### (.*)/gm, 'h3. $1')
    .replace(/\*\*(.*?)\*\*/g, '*$1*')
    .replace(/\*(.*?)\*/g, '_$1_')
    .replace(/`(.*?)`/g, '{{$1}}');
}

// Helper function to convert test cases to HTML format
function convertToHtmlFormat(testCases) {
  // Simple conversion to HTML
  return testCases
    .replace(/^# (.*)/gm, '<h1>$1</h1>')
    .replace(/^## (.*)/gm, '<h2>$1</h2>')
    .replace(/^### (.*)/gm, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// Export individual functions for the manifest
export const run = resolver.getDefinition('run');
export const getIssueDetails = resolver.getDefinition('getIssueDetails');
export const generateTestCasesDirectly = resolver.getDefinition('generateTestCasesDirectly');
export const exportTestCases = resolver.getDefinition('exportTestCases');
