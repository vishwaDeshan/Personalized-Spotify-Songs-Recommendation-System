# music-recommendation-system

This project consists of a FastAPI backend and a React frontend, which works together to provide personalized music recommendations based on user input and data from Spotify.

## Prerequisites

Ensure you have the following installed:

- Python 3.10+ (or compatible version)
- Node.js and npm (for React frontend)
- MongoDB (for storing data)
- Git (for cloning the repository)

## Backend Setup (FastAPI)

1. Navigate to the backend folder:

```bash
cd music-recommendation-backend
```

2. Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

   - **Windows**:

   ```bash
   .\venv\Scripts\activate
   ```

   - **Linux/Mac**:

   ```bash
   source venv/bin/activate
   ```

4. Install the required Python packages:

```bash
pip install -r requirements.txt

5. Run the FastAPI backend:

```bash
uvicorn app.main:app --reload
```

This will start the backend server at `http://127.0.0.1:8000`.


```plaintext
├── music-recommendation-backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI main application file
│   │   ├── routers/
│   │   │   ├── auth.py                # Routes for login/signup (Log In, Sign Up)
│   │   │   ├── recommendations.py     # Routes for song recommendations (Display Recommended Songs, Rate Songs)
│   │   │   └── health.py              # Health check routes
│   │   ├── models/                    # Pydantic models for data validation
│   │   │   └── user.py                # User data models (e.g., username, password, preferences)
│   │   ├── schemas/                   # Database models (SQLAlchemy)
│   │   │   └── user.py                # SQLAlchemy models for "Main DB"
│   │   ├── databases/
│   │   │   ├── main_db/
│   │   │   │   ├── schema.sql         # Database schema for user data, preferences, etc.
│   │   │   │   └── migrations/        # Database migration scripts (e.g., Alembic)
│   │   │   └── other_users_profiles/
│   │   │       ├── schema.sql         # Schema for other users' profiles dataset
│   │   │       └── migrations/        # Migration scripts for other users' dataset
│   │   ├── apis/
│   │   │   └── spotify_api.py         # Integration with Spotify API for genres and songs
│   │   ├── modules/
│   │   │   ├── similar_user_calculation.py  # Calculates similar users based on preferences
│   │   │   ├── optimal_point_calculation.py # Determines optimal song based on mood and similarity
│   │   │   ├── song_list_generator.py       # Generates recommended song list
│   │   │   └── song_regional_identifier.py  # Filters songs based on regional preferences
│   │   ├── ml/
│   │   │   └── rl_model.py               # Reinforcement learning model for optimizing recommendations
│   │   ├── config/
│   │   │   └── settings.py              # Configuration settings (e.g., database URL, API keys)
│   │   └── utils/                      # Utility functions (e.g., data processing)
│   ├── tests/                          # Unit and integration tests for backend modules
│   ├── .env                            # Environment variables (e.g., Spotify API keys, DB credentials)
│   └── requirements.txt                # Python dependencies
├── music-recommendation-frontend/
│   ├── public/
│   │   └── index.html                  # Static HTML file for the frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Login.js                # Login component
│   │   │   ├── Signup.js               # Signup component
│   │   │   ├── RecommendedSongs.js     # Component to display recommended songs
│   │   │   └── RateSongs.js            # Component for rating songs
│   │   ├── services/
│   │   │   └── api.js                  # API service to communicate with backend
│   │   └── App.js                      # Main React application file
│   ├── tests/                          # Unit and integration tests for frontend components
│   └── package.json                    # Frontend dependencies and scripts
├── .gitignore                          # Git ignore file
└── README.md                           # Project documentation
```
