# Setup Stripe billing

Guide pratique pour activer les paiements côté production.

## 1. Compte Stripe

Créé sur https://dashboard.stripe.com — organisation "Traduction Audio", compte
`traduction-audio.fr`. **Toujours vérifier en haut du dashboard qu'on est bien
sur ce compte** (pas sur `Tarasenko` ou `dev-ai`) avant chaque action.

## 2. Produits + prix

Dashboard → **Catalogue de produits** → **+ Ajouter un produit**.

Créer un seul produit `Pro` avec **deux prix récurrents** :

### Prix 1 — Mensuel
- Nom : `Pro mensuel`
- Type de tarification : **Récurrent**
- Montant : `9,90` EUR
- Période de facturation : **Tous les mois**
- ID récurrent : Stripe génère automatiquement un `price_...`

### Prix 2 — Annuel
- Nom : `Pro annuel`
- Type de tarification : **Récurrent**
- Montant : `99` EUR
- Période de facturation : **Tous les ans**
- ID récurrent : `price_...`

Une fois créés, copier les 2 IDs `price_...` et les mettre dans le `.env` du
serveur :
```
STRIPE_PRICE_PRO_MONTHLY=price_xxxxxxxxxxxxxxxxxxxxxx
STRIPE_PRICE_PRO_YEARLY=price_xxxxxxxxxxxxxxxxxxxxxx
```

## 3. Clés API

Dashboard → **Développeurs → Clés API**.

- **Clé publiable** (commence par `pk_live_` ou `pk_test_`) → à mettre dans
  `NEXT_PUBLIC_STRIPE_PUBLIC_KEY`. Safe à exposer côté client.
- **Clé secrète** (commence par `sk_live_` ou `sk_test_`) → à mettre dans
  `STRIPE_SECRET_KEY`. **Jamais dans git, jamais dans un chat, jamais côté
  frontend.**

Recommandation : **commencer en mode test** (`sk_test_...`, `pk_test_...`) pour
tester le flow sans facturation réelle. Basculer en `_live_` quand tout marche.

## 4. Webhook

Dashboard → **Développeurs → Webhooks → + Add endpoint**.

- URL : `https://traduction-audio.fr/api/billing/webhook`
- Description : `Traduction Audio production`
- Événements à sélectionner :
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.paid`
  - `invoice.payment_failed`
  - `checkout.session.completed`

Après création, Stripe affiche un **secret de signature** (`whsec_...`). Le
copier et le mettre dans le `.env` :
```
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 5. Portail client Stripe

Dashboard → **Paramètres → Portail client** (Customer Portal).

Cocher :
- ✅ Annulation d'abonnement (immédiate ou fin de période)
- ✅ Mise à jour du moyen de paiement
- ✅ Téléchargement des factures
- ✅ Mise à jour des coordonnées

Enregistrer la configuration.

## 6. Config sur le serveur

Sur hermes, éditer `~/traudio/.env` :

```bash
# Clés Stripe
STRIPE_SECRET_KEY=sk_live_...           # ou sk_test_... en dev
NEXT_PUBLIC_STRIPE_PUBLIC_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Prices IDs (depuis étape 2)
STRIPE_PRICE_PRO_MONTHLY=price_...
STRIPE_PRICE_PRO_YEARLY=price_...

# URL de base pour les redirections post-paiement
APP_BASE_URL=https://traduction-audio.fr
```

Puis :
```bash
docker compose build gateway frontend
docker compose up -d --force-recreate gateway frontend
```

## 7. Tester le flow

1. Se connecter sur `https://traduction-audio.fr`
2. Aller sur `/tarifs`
3. Cliquer "S'abonner mensuel" → redirection vers Stripe Checkout
4. En mode test, utiliser la carte `4242 4242 4242 4242`, expiration future, CVC 123
5. Après paiement, redirect vers `/tarifs/succes`
6. Vérifier côté Stripe dashboard : abonnement actif + webhook reçu (200 OK)
7. Vérifier côté DB Gateway : `user.subscription_tier = 'pro'`

## 8. Passage en mode LIVE

Une fois tout testé en `_test_` :
1. Récupérer les clés `_live_` dans le dashboard
2. Remplacer dans `.env` du serveur
3. **Recréer les prix en mode LIVE** (les prix créés en test ne sont PAS
   automatiquement copiés en live)
4. Mettre à jour `STRIPE_PRICE_PRO_MONTHLY` et `STRIPE_PRICE_PRO_YEARLY` avec
   les nouveaux `price_...` LIVE
5. **Recréer le webhook en mode LIVE** avec un nouveau `whsec_...`
6. Rebuild + restart

## 9. Après première vente réelle

Obligations administratives à activer :

- **Adhésion à un médiateur de la consommation** (obligatoire B2C, art. L616-1
  CCons) — CM2C ~80 €/an, mettre à jour `frontend/app/cgv/page.tsx` avec les
  coordonnées du médiateur
- **Compte bancaire pro** relié à Stripe pour recevoir les virements
- **Suivre les seuils TVA** — si le CA dépasse 36 800 €/an sur services, la
  franchise en base ne s'applique plus → mettre à jour `.env` et les mentions
  TVA dans le code (ligne "franchise en base" dans `/mentions-legales` et
  `/cgv` à retirer)

## Support

En cas de problème sur le webhook :
- Dashboard Stripe → Webhooks → Cliquer l'endpoint → **Envoyer un événement
  test** pour vérifier que le serveur répond bien 200
- Logs côté gateway : `docker compose logs gateway --tail 50 | grep billing`
