// Vercel serverless function to get AI explanation from Groq
export default async function handler(req, res) {
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { question, answer, section } = req.body;
    const groqApiKey = process.env.GROQ_API_KEY;

    console.log('🔧 Backend /api/explain called');
    console.log('📋 Request body:', { question: question?.substring(0, 50), answer, section });
    console.log('🔑 API Key exists:', !!groqApiKey);

    if (!groqApiKey) {
        console.error('❌ GROQ_API_KEY is not set in environment variables');
        return res.status(400).json({ error: 'Groq API key not configured in environment' });
    }

    if (!question || !answer) {
        console.error('❌ Missing required fields:', { question: !!question, answer: !!answer });
        return res.status(400).json({ error: 'Missing question or answer' });
    }

    try {
        console.log('🚀 Calling Groq API...');

        const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${groqApiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'mixtral-8x7b-32768',
                messages: [{
                    role: 'user',
                    content: `${section && section.includes('General') ? 'GK' : section || 'General'} Question - Provide EXAM SHORTCUT METHOD

Question: ${question}
Correct Answer: ${answer}

Give response in this format:
1. SHORTCUT: [Quick trick to solve]
2. WHY: [Why this answer works]
3. KEY POINT: [What to remember]

Be very concise - just the essentials for exam prep.`
                }],
                temperature: 0.7,
                max_tokens: 300
            })
        });

        const data = await groqResponse.json();

        console.log('📨 Groq API response status:', groqResponse.status);
        console.log('📨 Groq API response:', JSON.stringify(data).substring(0, 200));

        if (groqResponse.status !== 200) {
            console.error('❌ Groq API error:', data);
            return res.status(400).json({
                error: 'API Error: ' + (data.error?.message || 'Unknown error'),
                details: data
            });
        }

        if(data.choices && data.choices[0]) {
            console.log('✅ Explanation generated successfully');
            return res.status(200).json({
                explanation: data.choices[0].message.content
            });
        } else {
            console.error('❌ No choices in Groq response:', data);
            return res.status(400).json({
                error: 'No explanation generated',
                details: data
            });
        }
    } catch (error) {
        console.error('❌ Groq API Error:', error);
        return res.status(500).json({
            error: 'Failed to get explanation: ' + error.message,
            details: error.toString()
        });
    }
}
