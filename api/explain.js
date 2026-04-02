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

    const { question, answer, section, topic, options } = req.body;
    const groqApiKey = process.env.GROQ_API_KEY;

    console.log('🔧 Backend /api/explain called');
    console.log('📋 Request body:', { question: question?.substring(0, 50), answer, section, topic, optionsCount: options?.length });
    console.log('🔑 API Key exists:', !!groqApiKey);

    if (!groqApiKey) {
        console.error('❌ GROQ_API_KEY is not set in environment variables');
        return res.status(400).json({ error: 'Groq API key not configured in environment' });
    }

    if (!question || !options) {
        console.error('❌ Missing required fields:', { question: !!question, options: !!options });
        return res.status(400).json({ error: 'Missing question or options' });
    }

    try {
        console.log('🚀 Calling Groq API...');

        // Format options for the prompt
        let optionsText = '';
        if (options && Array.isArray(options)) {
            optionsText = options.map((opt, idx) => {
                const val = typeof opt === 'object' ? opt.text : opt;
                const key = typeof opt === 'object' ? opt.key : String.fromCharCode(65+idx);
                return `${key}. ${val}`;
            }).join('\n');
        }

        // Determine if we need to identify the answer or explain existing one
        const isAnswerProvided = answer && answer !== 'undefined';

        const userPrompt = isAnswerProvided
            ? `You are an AFCAT exam expert tutor.

QUESTION TYPE: ${topic || 'General'}
SUBJECT: ${section || 'General'}

QUESTION:
${question}

OPTIONS:
${optionsText}

CORRECT ANSWER: ${answer}

Provide an EXAM SHORTCUT explanation in this exact format:

1. SHORTCUT: [Quick trick/method to solve - explain why this answer is correct]
2. WHY IT WORKS: [Detailed explanation of why '${answer}' is the right choice]
3. KEY POINT: [One important thing to remember for similar questions]
4. EXAM TIP: [A quick tip to save time in the exam]

Be CONCISE and practical - just essentials for exam prep.`
            : `You are an AFCAT exam expert tutor. You MUST identify the correct answer.

QUESTION TYPE: ${topic || 'General'}
SUBJECT: ${section || 'General'}

QUESTION:
${question}

OPTIONS:
${optionsText}

Your task:
1. IDENTIFY which option (A, B, C, or D) is the CORRECT ANSWER
2. Provide the correct answer in format: CORRECT OPTION: [A/B/C/D]
3. Then explain using this format:

1. SHORTCUT: [Quick trick/method to solve]
2. WHY IT WORKS: [Why this option is correct]
3. KEY POINT: [One important thing to remember]
4. EXAM TIP: [A quick tip to save time]

Be CONCISE and practical.`;

        const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${groqApiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: 'llama-3.3-70b-versatile',
                messages: [{
                    role: 'user',
                    content: userPrompt
                }],
                temperature: 0.7,
                max_tokens: 400
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
