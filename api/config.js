// Vercel serverless function to serve API config
export default function handler(req, res) {
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    // Get API key from environment variable
    const groqApiKey = process.env.GROQ_API_KEY || '';

    return res.status(200).json({
        groq_api_key: groqApiKey,
        environment: process.env.NODE_ENV || 'production'
    });
}
