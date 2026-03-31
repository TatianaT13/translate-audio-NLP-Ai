from pydantic import BaseModel, EmailStr, field_validator


def _password_rules(v: str) -> str:
    if len(v) < 8:
        raise ValueError("Minimum 8 caractères")
    if not any(c.isupper() for c in v):
        raise ValueError("Au moins une majuscule requise")
    if not any(c.isdigit() for c in v):
        raise ValueError("Au moins un chiffre requis")
    return v


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _password_rules(v)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _password_rules(v)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        return _password_rules(v)


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"


class UserResponse(BaseModel):
    id:         int
    email:      str
    created_at: str

    class Config:
        from_attributes = True
