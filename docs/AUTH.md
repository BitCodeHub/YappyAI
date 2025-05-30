# Authentication System

AgenticSeek now includes JWT-based authentication to secure API endpoints.

## Overview

The authentication system provides:
- User registration and login
- JWT token-based authentication
- Protected API endpoints
- Token expiration (24 hours by default)

## Setup

1. Install the required dependencies:
```bash
pip install pyjwt bcrypt
```

2. Set the JWT secret key (optional):
```bash
export JWT_SECRET_KEY="your-very-secure-secret-key"
```

If not set, a default key will be used (change this in production!).

## API Endpoints

### Public Endpoints

#### Register a new user
```bash
POST /auth/register
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

#### Login
```bash
POST /auth/login
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

Response:
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer",
    "expires_in": 86400
}
```

### Protected Endpoints

All other endpoints now require authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your_access_token>
```

#### Get current user
```bash
GET /auth/me
Authorization: Bearer <token>
```

#### Query the AI
```bash
POST /query
Authorization: Bearer <token>
Content-Type: application/json

{
    "query": "Your question here"
}
```

## Usage Example

```python
import requests

# 1. Register
response = requests.post("http://localhost:8000/auth/register", 
    json={"username": "user1", "password": "pass123"})

# 2. Login
response = requests.post("http://localhost:8000/auth/login",
    json={"username": "user1", "password": "pass123"})
token = response.json()["access_token"]

# 3. Make authenticated requests
headers = {"Authorization": f"Bearer {token}"}
response = requests.post("http://localhost:8000/query",
    json={"query": "What is 2+2?"}, 
    headers=headers)
```

## Frontend Integration

Update your frontend to:
1. Store the token (localStorage or sessionStorage)
2. Include the token in all API requests
3. Handle 401 responses by redirecting to login

Example with axios:
```javascript
// Set default header
axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

// Or per request
axios.post('/query', data, {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```

## Security Notes

1. **Change the default secret key** in production
2. **Use HTTPS** in production to protect tokens in transit
3. **Store tokens securely** on the client side
4. **Implement token refresh** for better UX (optional)
5. **Add rate limiting** to prevent brute force attacks (optional)

## User Data Storage

User credentials are stored in `users.json` with bcrypt-hashed passwords.
In production, consider using a proper database instead.