// api/proxy.js

import fetch from 'node-fetch';  // Ensure node-fetch is installed
import { defineEventHandler, createError } from 'h3';  // Vercel's handler for serverless functions

// Use this function to fetch fixtures
export default defineEventHandler(async (event) => {
  try {
    const url = event.req.url || ''; // Get the URL (if you have query parameters)
    
    if (url.includes('fixtures')) {
      // Fetching Fixtures
      const response = await fetch('https://fantasy.premierleague.com/api/fixtures/');
      const data = await response.json();
      return data;  // Return the data in a serverless response
    }

    if (url.includes('teams')) {
      // Fetching Teams
      const response = await fetch('https://fantasy.premierleague.com/api/bootstrap-static/');
      const data = await response.json();
      return data;
    }

    // Default error handling if the URL doesn't match
    return createError({
      statusCode: 404,
      statusMessage: 'Not Found',
      message: 'Invalid API endpoint requested.'
    });
    
  } catch (error) {
    console.error('Error fetching data:', error);
    return createError({
      statusCode: 500,
      statusMessage: 'Internal Server Error',
      message: 'Failed to fetch data from the API.'
    });
  }
});
