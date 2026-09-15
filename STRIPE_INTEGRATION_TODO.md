# Stripe integration — remaining setup

This is the single source of truth for finishing the Stripe Checkout
integration. Everything code-side is done; what follows is configuration.

## Values to Replace

The Checkout Session call uses real values for the sample_only fields
(`mode`, `success_url`, `cancel_url` are hardcoded to the real production
paths of the app; only the `line_items[].price` reads its value from
environment variables that must be set to real Stripe Price IDs).

**Files containing placeholders:**
- [.env.example](.env.example) (lines 106-107)
- [backend/services/gateway/src/gateway/billing.py](backend/services/gateway/src/gateway/billing.py) (indirectly — reads Price IDs from env)

| Field | Current value (env) | What to set |
|-------|---------------------|-------------|
| `line_items[].price` (monthly) | `STRIPE_PRICE_PRO_MONTHLY=price_REPLACE_WITH_YOUR_MONTHLY_PRICE_ID` | Real Stripe Price ID for the monthly recurring price (starts with `price_`), copied from https://dashboard.stripe.com/products |
| `line_items[].price` (yearly)  | `STRIPE_PRICE_PRO_YEARLY=price_REPLACE_WITH_YOUR_YEARLY_PRICE_ID`  | Real Stripe Price ID for the yearly recurring price |

Setting these in `.env` on the server (not in git) is enough — the code
already reads them via `os.getenv(...)`.

## Configured Parameters

These parameters were fixed by the Checkout Studio UI and are already set
correctly in the code.

**Files containing these parameters:**
- [backend/services/gateway/src/gateway/billing.py](backend/services/gateway/src/gateway/billing.py) (around line 128)

| Parameter | Value |
|-----------|-------|
| `ui_mode` | `hosted` |
| `billing_address_collection` | `auto` |
| `phone_number_collection` | `{ "enabled": False }` |
| `automatic_tax` | `{ "enabled": False }` |
| `allow_promotion_codes` | `False` |
| `payment_method_collection` | `always` (mode is `subscription`) |
| `submit_type` | `auto` |
| `integration_identifier` | `hosted_web_0001` |
| `origin_context` | `web` |

## Notes on preserved parameters

The following parameters existed in the code before this update and were
preserved because they carry real business logic (they are not part of
Checkout Studio configuration):

- `customer=customer_id` — links the Checkout Session to an existing
  Stripe Customer created by `_ensure_customer()`. Prevents duplicate
  customer records across subscriptions for the same user.
- `metadata={"user_id": str(current_user.id)}` — used by the webhook
  handler to associate the resulting subscription events with the local
  user record in the database.

## SDK version note

This project uses the Python SDK (`stripe>=11.0`), which supports
`ui_mode` with value `"hosted"`. If the project is later upgraded to a
JS SDK version 21.0.0+, switch `ui_mode` to `"hosted_page"` (rule 10 of
the integration task).

## Environment variables — full list

Set these in `.env` on the server (Hetzner VPS `hermes`), never in git:

```bash
# Stripe API keys (get from https://dashboard.stripe.com/apikeys)
STRIPE_SECRET_KEY=sk_live_...           # server-only, never expose to browser
NEXT_PUBLIC_STRIPE_PUBLIC_KEY=pk_live_...  # safe to expose in browser bundle

# Webhook signing secret (get from https://dashboard.stripe.com/webhooks
# after creating an endpoint that points to /api/billing/webhook)
STRIPE_WEBHOOK_SECRET=whsec_...

# Price IDs (get from https://dashboard.stripe.com/products after creating
# the "Pro" product with two recurring prices)
STRIPE_PRICE_PRO_MONTHLY=price_...
STRIPE_PRICE_PRO_YEARLY=price_...

# Base URL of the app (used to build success_url / cancel_url)
APP_BASE_URL=https://traduction-audio.fr
```

## Setup steps (remaining, non-code)

1. **Create the product and prices in Stripe** — one product `Pro`, two
   recurring prices: monthly (`9.90 EUR` / month) and yearly (`99 EUR` /
   year). Copy the two `price_...` IDs into `.env`.

2. **Create a webhook endpoint** at
   `https://traduction-audio.fr/api/billing/webhook` with the following
   events subscribed:
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.paid`
   - `invoice.payment_failed`
   - `checkout.session.completed`
   
   Copy the signing secret (`whsec_...`) into `.env`.

3. **Enable the Customer Portal** in Stripe Dashboard → Settings →
   Customer portal. Check: cancel subscription, update payment method,
   download invoices.

4. **Complete Stripe KYC** (Managed Payments → "Informations requises"
   in the dashboard). Without this, real payments cannot be captured
   even if the code works. Provide ID, address proof, IBAN, and business
   info (SIRET, APE).

5. **Deploy on the server**:
   ```bash
   cd ~/traudio
   git fetch && git reset --hard origin/main
   nano .env    # paste the real Stripe values
   docker compose build gateway frontend
   docker compose up -d --force-recreate gateway frontend
   ```

## How the integration works — flow

1. User visits `/tarifs`, clicks "S'abonner mensuel" (or annual).
2. Frontend calls `POST /billing/checkout` with `{ plan: "monthly" }`.
3. Gateway creates a Stripe Customer if the user doesn't have one yet,
   then creates a hosted Checkout Session with the fixed_by_ui + real
   parameters and returns its `url`.
4. Frontend redirects the browser to that URL.
5. User pays on the Stripe-hosted page.
6. Stripe redirects to `/tarifs/succes?session_id=...`.
7. In parallel, Stripe fires a webhook to `/api/billing/webhook` with
   `checkout.session.completed` + `customer.subscription.created`.
8. The webhook handler updates the local `User` row:
   - `subscription_tier = "pro"`
   - `subscription_status = "active"`
   - `current_period_end = <timestamp from Stripe>`
9. Next time the user opens the app, `GET /billing/subscription` returns
   the new state.

## Testing

Use these Stripe test cards on the Checkout page while running with
`sk_test_...` keys:

| Card | Purpose |
|------|---------|
| `4242 4242 4242 4242` | Payment succeeds |
| `4000 0027 6000 3184` | 3D Secure authentication required |
| `4000 0000 0000 9995` | Card declined (insufficient funds) |

Expiration date: any future date. CVC: any 3 digits. ZIP: any 5 digits.

To test the webhook locally without a public URL, use the Stripe CLI:
```bash
stripe listen --forward-to localhost:8004/api/billing/webhook
```

## Next steps beyond this integration

- **Quota enforcement** — the DB fields `subscription_tier` / `_status`
  are set correctly, but no code yet blocks a `free` user from
  overusing `/process`. Add a middleware in the Pipeline service to
  return HTTP 402 Payment Required when `free` users exceed 10
  requests / month.
- **Invoicing** — Stripe already generates PDF invoices automatically
  and makes them available in the Customer Portal. No extra code
  needed unless you want to email them yourself.
- **Consumer mediation** — French law (art. L616-1 Consumer Code)
  requires B2C sellers to be affiliated with a consumer-mediation
  service before the first sale. Adhere to `CM2C` or equivalent
  (~80 €/year) and update `frontend/app/cgv/page.tsx` with the
  mediator's details.

## Resources

- Stripe Docs: https://docs.stripe.com/
- Stripe Support: https://support.stripe.com
- Stripe MCP: https://docs.stripe.com/mcp
