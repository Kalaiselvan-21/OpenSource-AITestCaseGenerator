import api, { route } from '@forge/api';
import { ApiClient } from './apiClient';
import config from '../config';
import { storeJob, getJob, updateJob, cleanupOldJobs } from './storage';

/**
 * Fetch issue details from JIRA API
 */
export const fetchIssueDetails = async (issueKey) => {
  try {
    console.log(`Making JIRA API request for issue: ${issueKey}`);
    
    const response = await api.asUser().requestJira(route`/rest/api/3/issue/${issueKey}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`JIRA API error: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`Failed to fetch issue: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('JIRA API response received');
    
    const summary = data.fields.summary || '';
    const description = data.fields.description ? extractTextFromADF(data.fields.description) : '';
    let acceptanceCriteria = '';
    
    // Try to find acceptance criteria in custom fields
    const customFields = Object.keys(data.fields).filter(key => key.startsWith('customfield_'));
    for (const field of customFields) {
      const fieldValue = data.fields[field];
      if (fieldValue && typeof fieldValue === 'string' && fieldValue.toLowerCase().includes('accept')) {
        acceptanceCriteria = fieldValue;
        break;
      }
    }
    
    // If no acceptance criteria found in custom fields, try to extract from description
    if (!acceptanceCriteria && description) {
      const acSection = extractAcceptanceCriteria(description);
      if (acSection) {
        acceptanceCriteria = acSection;
      }
    }
    
    return {
      summary,
      description,
      acceptanceCriteria
    };
  } catch (error) {
    console.error('Error fetching issue details:', error);
    throw error;
  }
};

/**
 * Start a test case generation job
 * This function returns immediately with a job ID
 */
export const startTestCaseGenerationJob = async (payload) => {
  try {
    console.log('Starting test case generation job');
    
    // Validate essential fields
    if (!payload.description || !payload.acceptance_criteria) {
      throw new Error('Missing required fields: description and acceptance_criteria');
    }
    
    // Generate a unique job ID
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    
    // Create job data
    const jobData = {
      status: 'pending',
      payload,
      startTime: Date.now(),
      timeoutAt: Date.now() + 60000 // 60 second timeout
    };
    
    // Store job in Forge storage
    await storeJob(jobId, jobData);
    
    // Start the job in the background (don't await)
    generateTestCasesAsync(jobId, payload);
    
    return {
      success: true,
      jobId
    };
  } catch (error) {
    console.error('Error starting test case generation job:', error);
    throw error;
  }
};

/**
 * Get the status of a test case generation job
 */
export const getJobStatus = async (jobId) => {
  try {
    // Get job from Forge storage
    const job = await getJob(jobId);
    
    if (!job) {
      return {
        success: false,
        error: 'Job not found'
      };
    }
    
    // Check if job has timed out
    if (job.status === 'processing' && job.timeoutAt && Date.now() > job.timeoutAt) {
      // Job has timed out, update status and return fallback test cases
      const fallbackTestCases = generateFallbackTestCases(job.payload);
      
      // Update job with timeout status and fallback results
      await updateJob(jobId, {
        ...job,
        status: 'completed',
        result: fallbackTestCases,
        timedOut: true,
        completionTime: Date.now()
      });
      
      return {
        success: true,
        status: 'completed',
        result: fallbackTestCases,
        timedOut: true
      };
    }
    
    // If the job is complete, return the result
    if (job.status === 'completed') {
      return {
        success: true,
        status: 'completed',
        result: job.result,
        timedOut: job.timedOut || false
      };
    }
    
    // If the job failed, return the error
    if (job.status === 'failed') {
      return {
        success: false,
        status: 'failed',
        error: job.error
      };
    }
    
    // If the job is still pending or processing, return the status
    return {
      success: true,
      status: job.status
    };
  } catch (error) {
    console.error('Error getting job status:', error);
    return {
      success: false,
      error: `Error getting job status: ${error.message}`
    };
  }
};

/**
 * Asynchronous function to generate test cases
 * This runs in the background and updates the job store when complete
 */
async function generateTestCasesAsync(jobId, payload) {
  try {
    console.log(`Processing job ${jobId}`);
    
    // Get the job from storage
    const job = await getJob(jobId);
    
    if (!job) {
      console.error(`Job ${jobId} not found in storage`);
      return;
    }
    
    // Update job status to processing
    await updateJob(jobId, {
      ...job,
      status: 'processing'
    });
    
    // Try to generate test cases with a timeout
    try {
      // Use a direct API call with a timeout
      const testCases = await generateDirectTestCases(payload);
      
      // Parse the test cases if they're in JSON format
      let parsedTestCases = testCases;
      try {
        // Check if the result is a JSON string
        if (typeof testCases === 'string' && testCases.trim().startsWith('{')) {
          const jsonResult = JSON.parse(testCases);
          
          // If it's a JSON object with test_cases property, format it nicely
          if (jsonResult.test_cases) {
            parsedTestCases = formatJsonTestCases(jsonResult.test_cases);
          }
        }
      } catch (parseError) {
        console.error('Error parsing test cases JSON:', parseError);
        // Continue with the original string if parsing fails
      }
      
      // Update the job with the result
      await updateJob(jobId, {
        ...job,
        status: 'completed',
        result: parsedTestCases,
        completionTime: Date.now()
      });
      
      console.log(`Job ${jobId} completed successfully`);
      
      // Update token usage stats after successful generation
      try {
        await updateTokenUsageStats(jobId);
      } catch (tokenError) {
        console.error(`Error updating token usage stats for job ${jobId}:`, tokenError);
      }
      
    } catch (timeoutError) {
      console.error(`Job ${jobId} timed out:`, timeoutError);
      
      // Generate fallback test cases
      const fallbackTestCases = generateFallbackTestCases(payload);
      
      // Update the job with fallback results
      await updateJob(jobId, {
        ...job,
        status: 'completed',
        result: fallbackTestCases,
        timedOut: true,
        completionTime: Date.now()
      });
      
      console.log(`Job ${jobId} completed with fallback test cases`);
    }
    
    // Clean up old jobs
    await cleanupOldJobs(10);
    
  } catch (error) {
    console.error(`Error processing job ${jobId}:`, error);
    
    try {
      // Get the job from storage
      const job = await getJob(jobId);
      
      if (job) {
        // Update the job with the error
        await updateJob(jobId, {
          ...job,
          status: 'failed',
          error: error.message,
          completionTime: Date.now()
        });
      }
    } catch (storageError) {
      console.error(`Error updating job ${jobId} with error:`, storageError);
    }
  }
}

/**
 * After generating test cases, update token usage stats
 */
async function updateTokenUsageStats(jobId) {
  try {
    console.log(`Updating token usage stats for job ${jobId}`);
    
    // Make a request to the token-usage endpoint
    const result = await ApiClient.getTokenUsage();
    
    if (result.success) {
      console.log('Token usage stats updated successfully');
      console.log('Token usage:', JSON.stringify(result.token_usage, null, 2));
      return true;
    } else {
      console.error('Error updating token usage stats:', result.error);
      return false;
    }
  } catch (error) {
    console.error('Exception updating token usage stats:', error);
    return false;
  }
}

/**
 * Format JSON test cases into a readable string
 */
function formatJsonTestCases(testCases) {
  if (!Array.isArray(testCases)) {
    return JSON.stringify(testCases, null, 2);
  }
  
  return testCases.map((tc, index) => {
    let formattedTestCase = `Test Case ${index + 1}: ${tc.title || 'Untitled'}\n`;
    
    if (tc.description) {
      formattedTestCase += `Description: ${tc.description}\n\n`;
    }
    
    if (tc.steps && Array.isArray(tc.steps)) {
      formattedTestCase += `Steps:\n`;
      tc.steps.forEach((step, stepIndex) => {
        formattedTestCase += `${stepIndex + 1}. ${step}\n`;
      });
      formattedTestCase += '\n';
    }
    
    if (tc.expected_results && Array.isArray(tc.expected_results)) {
      formattedTestCase += `Expected Results:\n`;
      tc.expected_results.forEach((result, resultIndex) => {
        formattedTestCase += `- ${result}\n`;
      });
    }
    
    return formattedTestCase;
  }).join('\n\n');
}

/**
 * Generate test cases directly with a timeout
 */
async function generateDirectTestCases(payload) {
  // Make a direct API call to the backend
  const url = `${config.backendUrl}/generate-test-cases`;
  
  console.log(`Making direct API call to: ${url}`);
  
  // Create the payload - send the full content without truncation
  const apiPayload = {
    description: payload.description,
    acceptance_criteria: payload.acceptance_criteria,
    use_knowledge: true,
    use_retrieval: true,
    quick_mode: false,
    track_tokens: true  // Add flag to explicitly track tokens
  };
  
  // Make the API call with a timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 second timeout
  
  try {
    console.log('Sending API request with payload:', JSON.stringify(apiPayload).substring(0, 200) + '...');
    
    const response = await api.fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(apiPayload),
      signal: controller.signal
    });
    
    // Clear the timeout
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API request failed: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`API request failed: ${response.status} ${response.statusText}\n${errorText}`);
    }
    
    const result = await response.json();
    console.log('API response received:', JSON.stringify(result).substring(0, 200) + '...');
    
    if (!result.success) {
      console.error('API returned error:', result.error || 'Unknown error');
      throw new Error(result.error || 'API request failed');
    }
    
    // Handle both possible response formats
    if (result.test_cases) {
      return result.test_cases;
    } else {
      // If the response doesn't have test_cases property, return a default format
      console.warn('API response missing test_cases property, using fallback format');
      return {
        test_cases: [{
          title: "API Response Format Error",
          description: "The API response format was unexpected. Please check the backend logs.",
          steps: ["Contact system administrator"],
          expected_results: ["Issue resolved"]
        }]
      };
    }
  } catch (error) {
    console.error('Error in generateDirectTestCases:', error);
    
    if (error.name === 'AbortError') {
      throw new Error('API request timed out');
    }
    
    // Return a formatted error as test cases to prevent UI from disappearing
    return {
      test_cases: [{
        title: "Error Generating Test Cases",
        description: `An error occurred: ${error.message}`,
        steps: ["Check backend server is running", "Check network connectivity", "Review logs for details"],
        expected_results: ["Issue resolved"]
      }]
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Generate fallback test cases that are specific to the user story
 */
function generateFallbackTestCases(payload) {
  console.log('Generating fallback test cases');
  
  const description = payload.description || '';
  const acceptanceCriteria = payload.acceptance_criteria || '';
  
  // Extract key information from the description
  const lines = description.split('\n');
  const title = lines[0] || 'Feature Test';
  
  // Extract key words from the description and acceptance criteria
  const keywords = extractKeywords(description + ' ' + acceptanceCriteria);
  const feature = keywords[0] || 'feature';
  
  // Generate test cases based on acceptance criteria
  const testCases = [];
  
  // Add a basic test case with specific details from the user story
  testCases.push(`Test Case 1: Verify ${feature} Basic Functionality
Description: This test case verifies the basic functionality described in the user story.

Steps:
1. Log in to the system as a user with appropriate permissions
2. Navigate to the ${feature} section
3. Verify that the ${keywords[1] || 'functionality'} is available
4. Attempt to use the ${feature} with valid inputs
5. Verify the results match the expected behavior

Expected Results:
- The ${feature} is accessible to authorized users
- The ${keywords[1] || 'functionality'} works as described in the user story
- The system displays appropriate feedback after the action is completed
`);
  
  // Clean and filter acceptance criteria lines
  const acLines = acceptanceCriteria.split('\n')
    .map(line => line.trim())
    .filter(line => {
      // Skip empty lines
      if (!line) return false;
      
      // Skip lines that are just placeholder characters
      if (line === '-' || line === '*' || line === '•' || line === '>' || line === '#') return false;
      
      // Skip very short lines that are likely just formatting
      if (line.length <= 2 && !line.match(/[a-zA-Z0-9]/)) return false;
      
      // Skip lines that are just bullet points with no content
      if ((line.startsWith('-') || line.startsWith('*') || line.startsWith('•')) && line.length <= 3) return false;
      
      return true;
    });
  
  // Add test cases based on filtered acceptance criteria
  acLines.forEach((line, index) => {
    const acKeywords = extractKeywords(line);
    const acFeature = acKeywords[0] || feature;
    const condition = acKeywords[1] || 'condition';
    const behavior = line.split(' ').slice(-3).join(' ');
    
    // Create a proper test case title following best practices
    const testCaseTitle = `Verify ${acFeature} ${condition} ${behavior}`;
    
    testCases.push(`
Test Case ${index + 2}: ${testCaseTitle}
Description: This test case verifies the acceptance criteria: "${line}"

Steps:
1. Log in to the system as a user with appropriate permissions
2. Navigate to the ${acFeature} section
3. Set up the test conditions for testing "${line}"
4. Perform the actions necessary to test this specific acceptance criteria
5. Verify that the ${acFeature} behaves as expected

Expected Results:
- The acceptance criteria "${line}" is fully satisfied
- The ${acFeature} behaves correctly under the test conditions
- The system provides appropriate feedback or results
`);
  });
  
  return testCases.join('\n');
}

/**
 * Extract keywords from text
 */
function extractKeywords(text) {
  // Remove common words and punctuation
  const cleanText = text.toLowerCase()
    .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '')
    .replace(/\s{2,}/g, ' ');
  
  // Split into words
  const words = cleanText.split(' ');
  
  // Filter out common words
  const commonWords = ['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'being', 'to', 'of', 'for', 'with', 'by', 'about', 
                      'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                      'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                      'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                      'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                      'just', 'don', 'should', 'now', 'that', 'this', 'these', 'those'];
  
  const keywords = words.filter(word => 
    word.length > 3 && !commonWords.includes(word)
  );
  
  // Return unique keywords
  return [...new Set(keywords)];
}

/**
 * Legacy function for backward compatibility
 */
export const generateTestCases = async (payload) => {
  try {
    console.log('Using legacy generateTestCases function');
    
    // Start a job
    const jobResult = await startTestCaseGenerationJob(payload);
    const jobId = jobResult.jobId;
    
    // Return a message immediately
    return {
      success: true,
      test_cases: `
Test case generation has started in the background.
Job ID: ${jobId}

Your detailed test cases are being generated and will be available shortly.
Please check back in a few moments.

In the meantime, here are some general test case guidelines:
1. Test the main functionality described in the user story
2. Test edge cases and error conditions
3. Test with different input values
4. Verify that all acceptance criteria are met
`
    };
    
  } catch (error) {
    console.error('Error in legacy generateTestCases function:', error);
    throw error;
  }
};

/**
 * Extract acceptance criteria from description
 */
function extractAcceptanceCriteria(description) {
  const lines = description.split('\n');
  const acLines = [];
  let inAcSection = false;
  
  for (const line of lines) {
    const lowerLine = line.toLowerCase();
    
    if (lowerLine.includes('acceptance criteria') || lowerLine.includes('ac:')) {
      inAcSection = true;
      continue;
    }
    
    if (inAcSection && line.trim()) {
      acLines.push(line);
    }
    
    if (inAcSection && (lowerLine.includes('implementation') || lowerLine.includes('notes:'))) {
      break;
    }
  }
  
  return acLines.join('\n');
}

/**
 * Extract text from Atlassian Document Format
 */
function extractTextFromADF(adf) {
  if (!adf) return '';
  if (typeof adf === 'string') return adf;
  
  let text = '';
  
  if (adf.content && Array.isArray(adf.content)) {
    for (const node of adf.content) {
      if (node.type === 'paragraph' && node.content) {
        text += extractNodeText(node) + '\n';
      } else if (node.type === 'bulletList' || node.type === 'orderedList') {
        text += extractListText(node);
      }
    }
  }
  
  return text.trim();
}

/**
 * Extract text from a node
 */
function extractNodeText(node) {
  if (!node.content) return '';
  
  return node.content
    .filter(content => content.type === 'text')
    .map(content => content.text)
    .join(' ');
}

/**
 * Extract text from a list
 */
function extractListText(listNode) {
  if (!listNode.content) return '';
  
  return listNode.content
    .filter(item => item.type === 'listItem')
    .map(item => '- ' + extractNodeText(item))
    .join('\n');
}
