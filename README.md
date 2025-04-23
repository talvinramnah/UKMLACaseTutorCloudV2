# UKMLA Case-Based Tutor

An interactive medical case study application for UKMLA preparation, powered by OpenAI and Supabase.

## Project Structure
```
.
├── UKMLACaseBasedTutor7Cloud.py  # Main application file
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── secrets.toml              # Configuration secrets
└── CaseFiles/                    # Directory containing case study files
    ├── Acute coronary syndrome.txt
    ├── Adult Advanced Life Support.txt
    └── ...
```

## Local Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your secrets in `.streamlit/secrets.toml`

3. Run the application:
```bash
streamlit run UKMLACaseBasedTutor7Cloud.py
```

## Deployment to Streamlit Cloud

1. Push your code to a GitHub repository

2. Visit [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repository

4. Add your secrets in the Streamlit Cloud dashboard:
   - OPENAI_API_KEY
   - OPENAI_ASSISTANT_ID
   - SUPABASE_URL
   - SUPABASE_KEY

5. Deploy!

## Important Notes

- Ensure all case files are present in the `CaseFiles` directory
- Keep your API keys and secrets secure
- Never commit `.streamlit/secrets.toml` to version control 