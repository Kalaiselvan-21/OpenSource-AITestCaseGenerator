/**
 * Configuration file for the AI Test Case Generator Forge app
 * Contains environment-specific settings for development and production
 */

export const config = {
  development: {
    backendUrl: 'https://e056-49-207-192-42.ngrok-free.app',
    apiEndpoints: {
      generateTestCases: '/generate-test-cases',
      health: '/health'
    }
  },
  production: {
    backendUrl: 'https://e056-49-207-192-42.ngrok-free.app',
    apiEndpoints: {
      generateTestCases: '/generate-test-cases',
      health: '/health'
    }
  }
};

// Determine environment based on Forge environment variable or default to development
const environment = process.env.FORGE_ENV === 'production' ? 'production' : 'development';

// Export the configuration for the current environment
export default config[environment];
