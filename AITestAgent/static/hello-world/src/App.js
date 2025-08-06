import React, { useEffect, useState } from 'react';
import { invoke } from '@forge/bridge';
import './App.css';

// Import view separately to avoid potential initialization issues
let view;
try {
  view = require('@forge/bridge').view;
} catch (e) {
  console.error('Error importing view from bridge:', e);
}

function App() {
  // State variables
  const [issueKey, setIssueKey] = useState(null);
  const [issueDetails, setIssueDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userStory, setUserStory] = useState('');
  const [acceptanceCriteria, setAcceptanceCriteria] = useState('');
  const [useKnowledge, setUseKnowledge] = useState(true);
  const [useRetrieval, setUseRetrieval] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [testCases, setTestCases] = useState('');
  const [exportFormat, setExportFormat] = useState('markdown');
  const [formattedTestCases, setFormattedTestCases] = useState('');
  const [debugInfo, setDebugInfo] = useState([]);
  const [showDebug, setShowDebug] = useState(false);
  
  // Add debug information with timestamp
  const addDebugInfo = (message) => {
    const timestamp = new Date().toISOString();
    setDebugInfo(prev => [...prev, `[${timestamp}] ${message}`]);
  };
  
  // Format test cases from JSON to readable format
  const formatTestCases = (testCasesJson) => {
    try {
      // Parse the JSON string if it's a string
      let testCases = testCasesJson;
      if (typeof testCasesJson === 'string') {
        try {
          testCases = JSON.parse(testCasesJson);
        } catch (e) {
          // If it's not valid JSON, return as is
          return testCasesJson;
        }
      }
      
      // If it's not an array, return as is
      if (!Array.isArray(testCases)) {
        return JSON.stringify(testCases, null, 2);
      }
      
      // Format each test case
      return testCases.map((tc, index) => {
        let formatted = `Test Case ${index + 1}: ${tc.title || 'Untitled'}\n`;
        formatted += `Description: ${tc.description || 'No description provided'}\n\n`;
        
        if (tc.steps && Array.isArray(tc.steps)) {
          formatted += `Steps:\n`;
          tc.steps.forEach((step, stepIndex) => {
            formatted += `${stepIndex + 1}. ${step}\n`;
          });
          formatted += '\n';
        }
        
        if (tc.expected_results && Array.isArray(tc.expected_results)) {
          formatted += `Expected Results:\n`;
          tc.expected_results.forEach((result) => {
            formatted += `- ${result}\n`;
          });
        }
        
        return formatted;
      }).join('\n---\n\n');
    } catch (error) {
      console.error('Error formatting test cases:', error);
      return 'Error formatting test cases: ' + error.message;
    }
  };
  
  // Effect to fetch issue details when the component mounts
  useEffect(() => {
    const fetchIssueDetails = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Get the context from the view
        if (view && view.getContext) {
          try {
            const context = await view.getContext();
            addDebugInfo(`Got context: ${JSON.stringify(context)}`);
            
            if (context && context.extension && context.extension.issue && context.extension.issue.key) {
              const key = context.extension.issue.key;
              addDebugInfo(`Found issue key in context: ${key}`);
              setIssueKey(key);
            } else {
              // Fallback to hardcoded issue for testing
              const testKey = 'SCRUM-1'; // Use a key that exists in your JIRA
              addDebugInfo(`No issue key found in context, using test key: ${testKey}`);
              setIssueKey(testKey);
            }
          } catch (contextError) {
            addDebugInfo(`Error getting context: ${contextError.message}`);
            
            // Fallback to hardcoded issue for testing
            const testKey = 'SCRUM-1'; // Use a key that exists in your JIRA
            addDebugInfo(`Error getting context, using test key: ${testKey}`);
            setIssueKey(testKey);
          }
        } else {
          addDebugInfo('View or getContext not available');
          
          // Fallback to hardcoded issue for testing
          const testKey = 'SCRUM-1'; // Use a key that exists in your JIRA
          addDebugInfo(`View not available, using test key: ${testKey}`);
          setIssueKey(testKey);
        }
      } catch (err) {
        addDebugInfo(`Error in fetchIssueDetails: ${err.message}`);
        setError(`Failed to get issue context: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };
    
    fetchIssueDetails();
  }, []);
  
  // Effect to fetch issue details when the issue key changes
  useEffect(() => {
    const getIssueDetails = async () => {
      if (!issueKey) return;
      
      setLoading(true);
      setError(null);
      
      try {
        addDebugInfo(`Fetching details for issue: ${issueKey}`);
        
        const result = await invoke('getIssueDetails', { issueKey });
        
        if (result.success) {
          addDebugInfo('Issue details fetched successfully');
          setIssueDetails(result.data);
          
          // Pre-fill the form with issue details
          setUserStory(result.data.description || '');
          setAcceptanceCriteria(result.data.acceptanceCriteria || '');
        } else {
          const errorMsg = `Failed to fetch issue details: ${result.error || 'Unknown error'}`;
          addDebugInfo(errorMsg);
          setError(errorMsg);
        }
      } catch (err) {
        const errorMsg = `Error fetching issue details: ${err.message}`;
        addDebugInfo(errorMsg);
        setError(errorMsg);
      } finally {
        setLoading(false);
      }
    };
    
    getIssueDetails();
  }, [issueKey]);
  
  // Handle export format change
  const handleExportFormatChange = (e) => {
    setExportFormat(e.target.value);
  };
  
  // Handle export button click
  const handleExport = async () => {
    if (!testCases) return;
    
    try {
      addDebugInfo(`Exporting test cases in ${exportFormat} format`);
      
      const result = await invoke('exportTestCases', {
        testCases,
        format: exportFormat
      });
      
      if (result.success) {
        setFormattedTestCases(result.formattedTestCases);
        addDebugInfo('Test cases exported successfully');
      } else {
        addDebugInfo(`Failed to export test cases: ${result.error || 'Unknown error'}`);
      }
    } catch (err) {
      addDebugInfo(`Error exporting test cases: ${err.message}`);
    }
  };
  
  // Handle generate test cases button click
  const handleGenerateTestCases = async () => {
    setGenerating(true);
    setError(null);
    
    // Always show a loading message in the test cases area to keep UI visible
    setTestCases(`Generating test cases...\n\nPlease wait while the AI analyzes your user story and generates relevant test cases.`);
    
    try {
      addDebugInfo('Generating test cases directly...');
      
      // Format the payload
      const payload = {
        description: userStory,
        acceptance_criteria: acceptanceCriteria,
        use_knowledge: useKnowledge,
        use_retrieval: useRetrieval
      };
      
      // Make a direct API call instead of using the job system
      const result = await invoke('genTestCases', payload);
      
      addDebugInfo(`Direct API call result: ${JSON.stringify(result).substring(0, 100)}...`);
      
      if (result && result.success) {
        // Check the type of test_cases and handle accordingly
        if (typeof result.test_cases === 'string') {
          try {
            // Try to parse it as JSON first
            const jsonData = JSON.parse(result.test_cases);
            
            // If it's an object with test_cases property, extract that
            if (jsonData && jsonData.test_cases && Array.isArray(jsonData.test_cases)) {
              const formattedTestCases = formatTestCases(jsonData.test_cases);
              setTestCases(formattedTestCases);
              addDebugInfo('Test cases extracted from JSON and formatted successfully');
            } else {
              // Otherwise format the whole object
              const formattedTestCases = formatTestCases(jsonData);
              setTestCases(formattedTestCases);
              addDebugInfo('JSON parsed and formatted successfully');
            }
          } catch (e) {
            // If it's not valid JSON, use it as is (it might be pre-formatted text)
            setTestCases(result.test_cases);
            addDebugInfo('Using pre-formatted test cases from backend');
          }
        } else if (typeof result.test_cases === 'object') {
          // If it's already an object, format it
          const formattedTestCases = formatTestCases(result.test_cases);
          setTestCases(formattedTestCases);
          addDebugInfo('Object test cases formatted successfully');
        } else {
          // Fallback for unexpected formats
          setTestCases(String(result.test_cases));
          addDebugInfo(`Converted test cases to string from: ${typeof result.test_cases}`);
        }
        
        // Add generation method info if available
        if (result.generation_method) {
          addDebugInfo(`Test cases generated using: ${result.generation_method}`);
        }
      } else {
        // Handle error but keep UI intact
        const errorMsg = result?.error || 'Unknown error generating test cases';
        addDebugInfo(`Error: ${errorMsg}`);
        setError(errorMsg);
        
        // Keep the test cases area visible with an error message
        setTestCases(`Error generating test cases:\n${errorMsg}\n\nPlease try again or contact support if the issue persists.`);
      }
    } catch (err) {
      // Handle exceptions but keep UI intact
      const errorMsg = err.message || 'Unknown error';
      addDebugInfo(`Exception: ${errorMsg}`);
      setError(`Failed to generate test cases: ${errorMsg}`);
      
      // Keep the test cases area visible with an error message
      setTestCases(`Error generating test cases:\n${errorMsg}\n\nPlease try again or contact support if the issue persists.`);
    } finally {
      setGenerating(false);
    }
  };

  // Fixed layout structure that never disappears
  return (
    <div style={{ padding: '16px' }}>
      <h2>AI Test Case Generator</h2>
      
      {/* Issue Details Section - Always visible */}
      <div style={{ marginBottom: '16px' }}>
        {loading ? (
          <p>Loading issue details...</p>
        ) : (
          <>
            <p>Issue Key: {issueKey || 'Unknown'}</p>
            {issueDetails && (
              <div>
                <h3>Issue Details:</h3>
                <p><strong>Summary:</strong> {issueDetails.summary}</p>
                <p><strong>Description:</strong> {issueDetails.description ? issueDetails.description.substring(0, 100) + '...' : 'No description'}</p>
                <p><strong>Acceptance Criteria:</strong> {issueDetails.acceptanceCriteria ? issueDetails.acceptanceCriteria.substring(0, 100) + '...' : 'No acceptance criteria'}</p>
              </div>
            )}
          </>
        )}
      </div>
      
      {/* Error Messages - Always in the same place */}
      {error && (
        <div style={{ 
          padding: '8px 16px', 
          backgroundColor: '#FFEBE6', 
          borderRadius: '4px',
          marginBottom: '16px',
          border: '1px solid #FF8F73'
        }}>
          <p style={{ color: '#DE350B', margin: '0' }}>{error}</p>
        </div>
      )}
      
      {/* Form Section - Always visible */}
      <div style={{ 
        marginTop: '24px', 
        marginBottom: '24px',
        padding: '16px',
        border: '1px solid #DFE1E6',
        borderRadius: '4px',
        backgroundColor: '#F4F5F7'
      }}>
        <h3>Generate Test Cases</h3>
        
        <div style={{ marginBottom: '16px' }}>
          <label htmlFor="userStory" style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
            User Story:
          </label>
          <textarea
            id="userStory"
            value={userStory}
            onChange={(e) => setUserStory(e.target.value)}
            style={{
              width: '100%',
              minHeight: '100px',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #DFE1E6'
            }}
            placeholder="Enter the user story here..."
          />
        </div>
        
        <div style={{ marginBottom: '16px' }}>
          <label htmlFor="acceptanceCriteria" style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
            Acceptance Criteria:
          </label>
          <textarea
            id="acceptanceCriteria"
            value={acceptanceCriteria}
            onChange={(e) => setAcceptanceCriteria(e.target.value)}
            style={{
              width: '100%',
              minHeight: '100px',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #DFE1E6'
            }}
            placeholder="Enter the acceptance criteria here..."
          />
        </div>
        
        <div style={{ marginBottom: '16px', display: 'flex', gap: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={useKnowledge}
              onChange={(e) => setUseKnowledge(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            Use Knowledge Base
          </label>
          
          <label style={{ display: 'flex', alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={useRetrieval}
              onChange={(e) => setUseRetrieval(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            Use Retrieval
          </label>
        </div>
        
        <button
          onClick={handleGenerateTestCases}
          disabled={generating || !userStory || !acceptanceCriteria}
          style={{
            backgroundColor: '#0052CC',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '8px 16px',
            cursor: generating || !userStory || !acceptanceCriteria ? 'not-allowed' : 'pointer',
            opacity: generating || !userStory || !acceptanceCriteria ? 0.7 : 1
          }}
        >
          {generating ? 'Generating...' : 'Generate Test Cases'}
        </button>
      </div>
      
      {/* Test Cases Output - Always visible, even if empty */}
      <div style={{ 
        marginTop: '24px',
        padding: '16px',
        border: '1px solid #DFE1E6',
        borderRadius: '4px',
        backgroundColor: '#FFFFFF'
      }}>
        <h3>Test Cases</h3>
        
        {/* Export Controls - Only visible when there are test cases */}
        {testCases && (
          <div style={{ marginBottom: '16px', display: 'flex', gap: '16px', alignItems: 'center' }}>
            <label style={{ display: 'flex', alignItems: 'center' }}>
              Export Format:
              <select
                value={exportFormat}
                onChange={handleExportFormatChange}
                style={{ marginLeft: '8px', padding: '4px' }}
              >
                <option value="markdown">Markdown</option>
                <option value="jira">JIRA</option>
                <option value="html">HTML</option>
              </select>
            </label>
            
            <button
              onClick={handleExport}
              style={{
                backgroundColor: '#0052CC',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '4px 12px',
                cursor: 'pointer'
              }}
            >
              Export
            </button>
          </div>
        )}
        
        {/* Test Cases Content - Always visible */}
        <pre style={{
          backgroundColor: '#f5f5f5',
          padding: '16px',
          borderRadius: '4px',
          whiteSpace: 'pre-wrap',
          overflowX: 'auto',
          border: '1px solid #DFE1E6',
          minHeight: '100px'
        }}>
          {testCases || 'No test cases generated yet. Click "Generate Test Cases" to create test cases.'}
        </pre>
      </div>
      
      {/* Formatted Test Cases - Only visible when there are formatted test cases */}
      {formattedTestCases && (
        <div style={{ 
          marginTop: '24px',
          padding: '16px',
          border: '1px solid #DFE1E6',
          borderRadius: '4px',
          backgroundColor: '#FFFFFF'
        }}>
          <h3>Formatted Test Cases ({exportFormat})</h3>
          <pre style={{
            backgroundColor: '#f5f5f5',
            padding: '16px',
            borderRadius: '4px',
            whiteSpace: 'pre-wrap',
            overflowX: 'auto',
            border: '1px solid #DFE1E6'
          }}>
            {formattedTestCases}
          </pre>
        </div>
      )}
      
      {/* Debug Information - Toggle visibility */}
      <div style={{ marginTop: '24px' }}>
        <button
          onClick={() => setShowDebug(!showDebug)}
          style={{
            backgroundColor: '#f5f5f5',
            border: '1px solid #DFE1E6',
            borderRadius: '4px',
            padding: '4px 12px',
            cursor: 'pointer'
          }}
        >
          {showDebug ? 'Hide Debug Info' : 'Show Debug Info'}
        </button>
        
        {showDebug && (
          <pre style={{
            backgroundColor: '#f5f5f5',
            padding: '16px',
            borderRadius: '4px',
            whiteSpace: 'pre-wrap',
            overflowX: 'auto',
            border: '1px solid #DFE1E6',
            marginTop: '8px',
            fontSize: '12px'
          }}>
            {debugInfo.join('\n')}
          </pre>
        )}
      </div>
    </div>
  );
}

export default App;
