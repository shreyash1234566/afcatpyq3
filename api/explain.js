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

    if (!groqApiKey) {
        return res.status(400).json({ error: 'Groq API key not configured' });
    }

    if (!question || !answer) {
        return res.status(400).json({ error: 'Missing question or answer' });
    }

    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${groqApiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'mixtral-8x7b-32768',
                messages: [{
                    role: 'user',
                    content: `${section.includes('General') ? 'GK' : section} Question - Provide EXAM SHORTCUT METHOD

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

        const data = await response.json();

        if (data.choices && data.choices[0]) {
            return res.status(200).json({
                explanation: data.choices[0].message.content
            });
        } else {
            return res.status(400).json({
                error: 'No explanation generated',
                details: data
            });
        }
    } catch (error) {
        console.error('Groq API Error:', error);
        return res.status(500).json({
            error: 'Failed to get explanation',
            details: error.message
        });
    }
}
