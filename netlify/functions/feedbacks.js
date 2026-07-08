const https = require('https');

// Helper to make HTTPS requests using standard Node.js module
function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = https.request(url, options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers,
          body: data
        });
      });
    });
    
    req.on('error', (err) => reject(err));
    
    if (options.body) {
      req.write(options.body);
    }
    req.end();
  });
}

exports.handler = async (event, context) => {
  const method = event.httpMethod;
  const pathParts = event.path.split('/').filter(Boolean);
  
  // Extract ID if present (e.g. from /api/feedbacks/xyz)
  const lastPart = pathParts.length > 0 ? pathParts[pathParts.length - 1] : null;
  const submissionId = lastPart && lastPart !== 'feedbacks' ? lastPart : null;

  // 1. Handle fallback POST (so AJAX requests on Netlify don't fail)
  if (method === 'POST') {
    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ success: true, note: 'Submitted via Netlify Forms' })
    };
  }

  // 2. Validate Dashboard Access Passcode
  const authHeader = event.headers['authorization'] || '';
  const expectedPassword = process.env.DASHBOARD_PASSWORD || 'admin123';

  if (authHeader !== expectedPassword) {
    return {
      statusCode: 401,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Unauthorized' })
    };
  }

  const siteId = process.env.NETLIFY_SITE_ID;
  const accessToken = process.env.NETLIFY_ACCESS_TOKEN;

  if (!siteId || !accessToken) {
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        error: 'Netlify environment variables (NETLIFY_SITE_ID and NETLIFY_ACCESS_TOKEN) are not configured.' 
      })
    };
  }

  // 3. GET: Retrieve submissions from Netlify API
  if (method === 'GET') {
    try {
      const url = `https://api.netlify.com/api/v1/sites/${siteId}/submissions`;
      const response = await makeRequest(url, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (response.statusCode !== 200) {
        return {
          statusCode: response.statusCode,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ error: `Netlify API error: ${response.body}` })
        };
      }

      const submissions = JSON.parse(response.body);

      // Transform Netlify submissions to match the format used by the frontend dashboard
      const formattedSubmissions = submissions.map(sub => ({
        id: sub.id,
        email: sub.data.email || 'anonymous',
        message: sub.data.message || '',
        timestamp: sub.created_at
      }));

      return {
        statusCode: 200,
        headers: { 
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify(formattedSubmissions)
      };
    } catch (error) {
      console.error('Error fetching submissions from Netlify:', error);
      return {
        statusCode: 500,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: 'Failed to retrieve feedback data.' })
      };
    }
  }

  // 4. DELETE: Delete a submission via Netlify API
  if (method === 'DELETE') {
    if (!submissionId) {
      return {
        statusCode: 400,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: 'Submission ID is required.' })
      };
    }

    try {
      const url = `https://api.netlify.com/api/v1/submissions/${submissionId}`;
      const response = await makeRequest(url, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        }
      });

      if (response.statusCode >= 300) {
        return {
          statusCode: response.statusCode,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ error: `Netlify API delete error: ${response.body}` })
        };
      }

      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ success: true })
      };
    } catch (error) {
      console.error('Error deleting submission on Netlify:', error);
      return {
        statusCode: 500,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: 'Failed to delete submission.' })
      };
    }
  }

  // Fallback for unsupported methods
  return {
    statusCode: 405,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ error: 'Method Not Allowed' })
  };
};
