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
    is_admin:   bool
    created_at: str

    class Config:
        from_attributes = True


class AdminUserResponse(BaseModel):
    id:         int
    email:      str
    is_active:  bool
    is_admin:   bool
    created_at: str | None

    class Config:
        from_attributes = True


class AdminUserUpdate(BaseModel):
    is_active: bool | None = None
    is_admin:  bool | None = None


class AdminStatsResponse(BaseModel):
    total_users:  int
    active_users: int
    admin_users:  int


class LangfuseScorePoint(BaseModel):
    name:    str
    value:   float
    comment: str | None = None


class LangfuseModelStat(BaseModel):
    whisper:         str
    llm:             str
    prompt_version:  str
    count:           int
    avg_total_ms:    float
    avg_stt_ms:      float
    avg_llm_ms:      float
    avg_bleu:        float | None = None


class LangfuseMetricsResponse(BaseModel):
    connected:          bool
    error:              str | None = None
    total_traces:       int = 0
    avg_total_ms:       float = 0
    avg_stt_ms:         float = 0
    avg_llm_ms:         float = 0
    avg_language_prob:  float = 0
    avg_bleu:           float = 0
    bleu_scores:        list[float] = []
    language_probs:     list[float] = []
    latencies_total:    list[float] = []
    model_stats:        list[LangfuseModelStat] = []
