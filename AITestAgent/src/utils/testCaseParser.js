/**
 * Utility functions for parsing test cases
 */

/**
 * Parse test cases from string to structured format
 * @param {string} testCasesString - The raw test cases string from the API
 * @returns {Array} - Array of structured test case objects
 */
export const parseTestCases = (testCasesString) => {
  if (!testCasesString) {
    return [];
  }
  
  console.log('Parsing test cases from string:', testCasesString.substring(0, 100) + '...');
  
  // First, try to parse markdown-style test cases (with ** or ## headers)
  const markdownTestCases = parseMarkdownTestCases(testCasesString);
  if (markdownTestCases.length > 0) {
    console.log(`Parsed ${markdownTestCases.length} test cases using markdown parser`);
    return markdownTestCases;
  }
  
  // If markdown parsing fails, try the original regex pattern
  const testCases = [];
  const regex = /(\d+\.\s*Title:\s*)(.*?)(\n\s*-\s*(?:Test )?Steps?:)([\s\S]*?)(\n\s*-\s*Expected Result:)([\s\S]*?)(?=\n\d+\.\s*Title:|\s*$)/g;
  
  let match;
  while ((match = regex.exec(testCasesString)) !== null) {
    const title = match[2].trim();
    const stepsText = match[4].trim();
    const expectedResult = match[6].trim();
    
    // Parse steps
    const steps = stepsText.split('\n')
      .map(step => step.trim())
      .filter(Boolean)
      .map(step => {
        // Remove bullet points or numbering if present
        return step.replace(/^[•\-\*\d]+[\.\)]\s*/, '');
      });
    
    testCases.push({
      title,
      steps,
      expectedResult
    });
  }
  
  console.log(`Parsed ${testCases.length} test cases using regex parser`);
  
  // If both parsing methods fail, try a more generic approach to extract any test cases
  if (testCases.length === 0) {
    console.log('Both parsers failed, trying generic extraction');
    return extractGenericTestCases(testCasesString);
  }
  
  return testCases;
};

/**
 * Parse markdown-formatted test cases
 * @param {string} testCasesString - The raw test cases string from the API
 * @returns {Array} - Array of structured test case objects
 */
function parseMarkdownTestCases(testCasesString) {
  const testCases = [];
  
  // First, remove any introductory text before the first test case
  const cleanedString = testCasesString.replace(/^.*?(?=\*\*Test Case|\d+\.\s*Title:)/s, '');
  
  // Match markdown-style test cases with ** or ## headers
  const testCaseBlocks = cleanedString.split(/\*\*Test Case \d+:|##\s*Test Case \d+:|^\d+\.\s*Title:/m);
  
  // Skip the first element if it's empty (split might return an empty string at the beginning)
  const startIndex = testCaseBlocks[0].trim() === '' ? 1 : 0;
  
  for (let i = startIndex; i < testCaseBlocks.length; i++) {
    const block = testCaseBlocks[i].trim();
    if (!block) continue;
    
    // Extract title
    let title = '';
    const titleMatch = block.match(/^([^*#\n]+)(?:\*\*|\n|$)/);
    if (titleMatch) {
      title = titleMatch[1].trim();
      // Remove any numbering from the title if present
      title = title.replace(/^\d+\.\s*/, '');
      // Remove "Title:" prefix if present
      title = title.replace(/^Title:\s*/, '');
    }
    
    // Extract steps
    const steps = [];
    const stepsSection = block.match(/(?:Test Steps?|Steps?):?\s*([\s\S]*?)(?:Expected Result|$)/i);
    if (stepsSection && stepsSection[1]) {
      const stepLines = stepsSection[1].split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.match(/^(?:Test Steps?|Steps?):?$/i));
      
      stepLines.forEach(line => {
        // Remove bullet points, numbers, or tabs
        const cleanedStep = line.replace(/^[•\-\*\t\d]+[\.\)]\s*/, '');
        if (cleanedStep) {
          steps.push(cleanedStep);
        }
      });
    }
    
    // Extract expected result
    let expectedResult = '';
    const resultSection = block.match(/Expected Result:?\s*([\s\S]*?)(?=\*\*Test Case|\*\*\d+\.|\n\s*\n\s*\d+\.|\n\s*\n\s*\*\*|$)/i);
    if (resultSection && resultSection[1]) {
      expectedResult = resultSection[1].trim()
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.match(/^Expected Result:?$/i))
        .join('\n');
    }
    
    if (title) {
      testCases.push({
        title,
        steps,
        expectedResult
      });
    }
  }
  
  return testCases;
}

/**
 * Extract test cases using a more generic approach
 * @param {string} testCasesString - The raw test cases string from the API
 * @returns {Array} - Array of structured test case objects
 */
function extractGenericTestCases(testCasesString) {
  const testCases = [];
  
  // Try to identify test case blocks by looking for patterns
  const possibleTestCaseBlocks = testCasesString.split(/\n\s*\n/);
  
  for (let block of possibleTestCaseBlocks) {
    block = block.trim();
    if (!block) continue;
    
    // Skip introductory text
    if (block.match(/^(Sure!|Here are|I'll create|Based on)/i)) continue;
    
    // Try to extract title, steps, and expected results
    let title = '';
    let steps = [];
    let expectedResult = '';
    
    // Extract title - look for the first line or a line with "Test Case" or similar
    const titleMatch = block.match(/^(?:\d+\.\s*)?(?:Test Case:?\s*)?([^\n]+)/i);
    if (titleMatch) {
      title = titleMatch[1].trim();
    }
    
    // Extract steps - look for lines with step indicators
    const stepMatches = block.match(/(?:Steps?|Test Steps?):?\s*([\s\S]*?)(?:Expected Result|$)/i);
    if (stepMatches && stepMatches[1]) {
      steps = stepMatches[1].trim()
        .split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.match(/^(?:Steps?|Test Steps?):?$/i))
        .map(line => line.replace(/^[•\-\*\t\d]+[\.\)]\s*/, ''));
    }
    
    // Extract expected results - look for lines with "Expected Result" or similar
    const resultMatches = block.match(/Expected Result:?\s*([\s\S]*?)$/i);
    if (resultMatches && resultMatches[1]) {
      expectedResult = resultMatches[1].trim();
    }
    
    // If we have at least a title, consider it a test case
    if (title) {
      testCases.push({
        title,
        steps: steps.length > 0 ? steps : ['No specific steps provided'],
        expectedResult: expectedResult || 'No specific expected result provided'
      });
    }
  }
  
  console.log(`Extracted ${testCases.length} test cases using generic extraction`);
  return testCases;
}

/**
 * Format test cases for display or export
 * @param {Array} testCases - Array of parsed test case objects
 * @param {string} format - Format type ('text', 'markdown', 'html')
 * @returns {string} - Formatted test cases
 */
export const formatTestCases = (testCases, format = 'text') => {
  if (!testCases || testCases.length === 0) {
    return '';
  }
  
  switch (format) {
    case 'markdown':
      return testCases.map((tc, index) => {
        const steps = tc.steps.map((step, i) => `   ${i + 1}. ${step}`).join('\n');
        return `## ${index + 1}. ${tc.title}\n\n### Test Steps:\n${steps}\n\n### Expected Result:\n${tc.expectedResult}`;
      }).join('\n\n');
      
    case 'html':
      return testCases.map((tc, index) => {
        const steps = tc.steps.map(step => `<li>${step}</li>`).join('');
        return `<div class="test-case">
          <h3>${index + 1}. ${tc.title}</h3>
          <h4>Test Steps:</h4>
          <ol>${steps}</ol>
          <h4>Expected Result:</h4>
          <p>${tc.expectedResult}</p>
        </div>`;
      }).join('');
      
    case 'text':
    default:
      return testCases.map((tc, index) => {
        const steps = tc.steps.map((step, i) => `   ${i + 1}. ${step}`).join('\n');
        return `${index + 1}. ${tc.title}\n   Test Steps:\n${steps}\n   Expected Result: ${tc.expectedResult}`;
      }).join('\n\n');
  }
};
