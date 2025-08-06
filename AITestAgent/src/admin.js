import Resolver from '@forge/resolver';
import api, { route } from '@forge/api';
import config from './config';

const resolver = new Resolver();

resolver.define('getAppInfo', async (req) => {
  return {
    name: 'AI Test Case Generator',
    version: '1.0.0',
    description: 'Generate test cases from user stories and acceptance criteria'
  };
});

resolver.define('testApiConnection', async (req) => {
  try {
    const healthEndpoint = `${config.backendUrl}${config.apiEndpoints.health}`;
    console.log(`Testing API connection to: ${healthEndpoint}`);
    
    const response = await api.fetch(healthEndpoint);
    
    if (!response.ok) {
      return {
        success: false,
        message: `API connection failed: ${response.status} ${response.statusText}`
      };
    }
    
    const data = await response.json();
    return {
      success: true,
      message: 'API connection successful',
      data: data
    };
  } catch (error) {
    return {
      success: false,
      message: `API connection error: ${error.message}`
    };
  }
});

export const handler = resolver.getDefinitions();
