import api from '@forge/api';
import config from '../config';

/**
 * API Client for making requests to the backend API
 */
export class ApiClient {
  /**
   * Make a GET request to the API
   * @param {string} endpoint - The API endpoint
   * @returns {Promise<Object>} - The API response
   */
  static async get(endpoint) {
    try {
      const url = `${config.backendUrl}${endpoint}`;
      console.log(`Making GET request to: ${url}`);
      
      const response = await api.fetch(url);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`API error: ${response.status} ${response.statusText}`, errorText);
        throw new Error(`API request failed: ${response.status} ${response.statusText}\n${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error making GET request:', error);
      throw error;
    }
  }
  
  /**
   * Make a POST request to the API
   * @param {string} endpoint - The API endpoint
   * @param {Object} data - The request payload
   * @returns {Promise<Object>} - The API response
   */
  static async post(endpoint, data) {
    try {
      const url = `${config.backendUrl}${endpoint}`;
      console.log(`Making POST request to: ${url}`);
      console.log('Request payload:', JSON.stringify(data, null, 2));
      
      // Validate payload before sending
      if (!data.description) {
        throw new Error('Missing required field: description');
      }
      
      if (!data.acceptance_criteria) {
        throw new Error('Missing required field: acceptance_criteria');
      }
      
      // Ensure the payload has the correct format
      const validatedPayload = {
        description: data.description,
        acceptance_criteria: data.acceptance_criteria,
        use_knowledge: data.use_knowledge !== false,
        use_retrieval: data.use_retrieval !== false
      };
      
      // Add optional fields if present
      if (data.quick_mode !== undefined) {
        validatedPayload.quick_mode = data.quick_mode;
      }
      
      if (data.structure_only !== undefined) {
        validatedPayload.structure_only = data.structure_only;
      }
      
      if (data.test_case_structure) {
        validatedPayload.test_case_structure = data.test_case_structure;
      }
      
      console.log('Validated payload:', JSON.stringify(validatedPayload, null, 2));
      
      const response = await api.fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(validatedPayload)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`API error: ${response.status} ${response.statusText}`, errorText);
        
        // Try to parse the error response if it's JSON
        try {
          const errorJson = JSON.parse(errorText);
          if (errorJson.error) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}\n${errorJson.error}`);
          }
        } catch (parseError) {
          // If parsing fails, just use the raw error text
        }
        
        throw new Error(`API request failed: ${response.status} ${response.statusText}\n${errorText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error making POST request:', error);
      throw error;
    }
  }
  
  /**
   * Get token usage statistics
   * @returns {Promise<Object>} - Token usage statistics
   */
  static async getTokenUsage() {
    return await this.get('/token-usage');
  }
}
