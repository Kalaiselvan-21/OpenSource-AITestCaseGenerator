/**
 * Utility functions for exporting test cases in different formats
 */

/**
 * Export test cases to different formats
 * @param {string} testCases - The test cases to export
 * @param {string} format - The format to export to (markdown, jira, html)
 * @returns {string} - The formatted test cases
 */
export function exportTestCases(testCases, format = 'markdown') {
  if (!testCases) {
    return '';
  }
  
  // If testCases is already a string, try to parse it as JSON
  let testCaseData = testCases;
  if (typeof testCases === 'string') {
    try {
      // Check if it's a JSON string
      if (testCases.trim().startsWith('{') || testCases.trim().startsWith('[')) {
        testCaseData = JSON.parse(testCases);
      }
    } catch (e) {
      // If parsing fails, keep the original string
      console.log('Failed to parse test cases as JSON, using as-is');
    }
  }
  
  // Handle different formats
  switch (format.toLowerCase()) {
    case 'markdown':
      return formatAsMarkdown(testCaseData);
    case 'jira':
      return formatAsJira(testCaseData);
    case 'html':
      return formatAsHtml(testCaseData);
    default:
      return formatAsMarkdown(testCaseData);
  }
}

/**
 * Format test cases as Markdown
 */
function formatAsMarkdown(testCases) {
  if (typeof testCases === 'string') {
    // If it's already a string and not JSON, return as is
    return testCases;
  }
  
  if (Array.isArray(testCases)) {
    // If it's an array of test cases
    return testCases.map((tc, index) => {
      let markdown = `## Test Case ${index + 1}: ${tc.title || 'Untitled'}\n\n`;
      
      if (tc.description) {
        markdown += `**Description:** ${tc.description}\n\n`;
      }
      
      if (tc.steps && Array.isArray(tc.steps)) {
        markdown += `### Steps:\n`;
        tc.steps.forEach((step, stepIndex) => {
          markdown += `${stepIndex + 1}. ${step}\n`;
        });
        markdown += '\n';
      }
      
      if (tc.expected_results && Array.isArray(tc.expected_results)) {
        markdown += `### Expected Results:\n`;
        tc.expected_results.forEach((result, resultIndex) => {
          markdown += `- ${result}\n`;
        });
      }
      
      return markdown;
    }).join('\n\n');
  } else if (testCases.test_cases && Array.isArray(testCases.test_cases)) {
    // If it's an object with a test_cases array
    return formatAsMarkdown(testCases.test_cases);
  } else {
    // If it's some other object, stringify it
    return '```json\n' + JSON.stringify(testCases, null, 2) + '\n```';
  }
}

/**
 * Format test cases as JIRA markup
 */
function formatAsJira(testCases) {
  if (typeof testCases === 'string') {
    // If it's already a string and not JSON, return as is
    return testCases;
  }
  
  if (Array.isArray(testCases)) {
    // If it's an array of test cases
    return testCases.map((tc, index) => {
      let jira = `h2. Test Case ${index + 1}: ${tc.title || 'Untitled'}\n\n`;
      
      if (tc.description) {
        jira += `*Description:* ${tc.description}\n\n`;
      }
      
      if (tc.steps && Array.isArray(tc.steps)) {
        jira += `h3. Steps:\n`;
        tc.steps.forEach((step, stepIndex) => {
          jira += `# ${step}\n`;
        });
        jira += '\n';
      }
      
      if (tc.expected_results && Array.isArray(tc.expected_results)) {
        jira += `h3. Expected Results:\n`;
        tc.expected_results.forEach((result, resultIndex) => {
          jira += `* ${result}\n`;
        });
      }
      
      return jira;
    }).join('\n\n');
  } else if (testCases.test_cases && Array.isArray(testCases.test_cases)) {
    // If it's an object with a test_cases array
    return formatAsJira(testCases.test_cases);
  } else {
    // If it's some other object, stringify it
    return '{code:json}\n' + JSON.stringify(testCases, null, 2) + '\n{code}';
  }
}

/**
 * Format test cases as HTML
 */
function formatAsHtml(testCases) {
  if (typeof testCases === 'string') {
    // If it's already a string and not JSON, wrap in pre tags
    return `<pre>${escapeHtml(testCases)}</pre>`;
  }
  
  if (Array.isArray(testCases)) {
    // If it's an array of test cases
    return testCases.map((tc, index) => {
      let html = `<h2>Test Case ${index + 1}: ${escapeHtml(tc.title || 'Untitled')}</h2>\n\n`;
      
      if (tc.description) {
        html += `<p><strong>Description:</strong> ${escapeHtml(tc.description)}</p>\n\n`;
      }
      
      if (tc.steps && Array.isArray(tc.steps)) {
        html += `<h3>Steps:</h3>\n<ol>`;
        tc.steps.forEach((step) => {
          html += `<li>${escapeHtml(step)}</li>\n`;
        });
        html += '</ol>\n\n';
      }
      
      if (tc.expected_results && Array.isArray(tc.expected_results)) {
        html += `<h3>Expected Results:</h3>\n<ul>`;
        tc.expected_results.forEach((result) => {
          html += `<li>${escapeHtml(result)}</li>\n`;
        });
        html += '</ul>';
      }
      
      return html;
    }).join('\n\n');
  } else if (testCases.test_cases && Array.isArray(testCases.test_cases)) {
    // If it's an object with a test_cases array
    return formatAsHtml(testCases.test_cases);
  } else {
    // If it's some other object, stringify it
    return `<pre><code>${escapeHtml(JSON.stringify(testCases, null, 2))}</code></pre>`;
  }
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
  if (typeof text !== 'string') {
    return '';
  }
  
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Export the utility class
export const ExportUtils = {
  exportTestCases
};
