"""
PDAX REST API client with HMAC-SHA384 authentication.

Authentication scheme (per PDAX docs):
  Access-Key       : your public API key
  Access-Nonce     : current timestamp in milliseconds (string)
  Access-Signature : HMAC-SHA384( nonce + full_url + body, secret_key )

All private endpoints require these three headers.
Public endpoints (ticker, order_book) work without authentication.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

from config import cfg

logger = logging.getLogger(__name__)


class PDAXClient:
    """Thin wrapper around the PDAX REST API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Auth helpers ───────────────────────────────────────────────────────────

    def _sign(self, nonce: str, url: str, body: str = "") -> str:
        message = (nonce + url + body).encode("utf-8")
        return hmac.new(cfg.secret_key.encode("utf-8"), message, hashlib.sha384).hexdigest()

    def _auth_headers(self, url: str, body: str = "") -> Dict[str, str]:
        nonce = str(int(time.time() * 1000))
        return {
            "Access-Key": cfg.api_key,
            "Access-Nonce": nonce,
            "Access-Signature": self._sign(nonce, url, body),
        }

    # ── Internal request dispatcher ────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        payload: Optional[Dict] = None,
        authenticated: bool = True,
    ) -> Any:
        url = cfg.base_url + path
        if params:
            url += "?" + urlencode(params)

        body = json.dumps(payload) if payload else ""
        headers = self._auth_headers(url, body) if authenticated else {}

        try:
            resp = self.session.request(
                method,
                url,
                headers=headers,
                data=body if body else None,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error("HTTP %s %s → %s: %s", method, path, resp.status_code, resp.text)
            raise
        except requests.RequestException as e:
            logger.error("Request error %s %s: %s", method, path, e)
            raise

    # ── Public endpoints ───────────────────────────────────────────────────────

    def get_ticker(self) -> Dict:
        """Return ticker data for all pairs."""
        return self._request("GET", "/ticker", authenticated=False)

    def get_order_book(self, pair: str) -> Dict:
        """
        Return order book for a single pair.
        pair: e.g. 'BTCPHP'
        Returns: { bids: [[price, qty], ...], asks: [[price, qty], ...] }
        """
        return self._request("GET", f"/order_book/{pair}", authenticated=False)

    # ── Account endpoints ──────────────────────────────────────────────────────

    def get_user(self) -> Dict:
        """Return user profile and wallet balances."""
        return self._request("GET", "/user")

    def get_balances(self) -> Dict[str, float]:
        """
        Convenience: return {currency: available_balance} dict.
        Example: {'PHP': 50000.0, 'BTC': 0.001, ...}
        """
        user = self.get_user()
        wallets = user.get("wallets", [])
        return {w["currency"]: float(w.get("available_balance", 0)) for w in wallets}

    # ── Order management ───────────────────────────────────────────────────────

    def place_limit_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: float,
        custom_id: Optional[str] = None,
    ) -> Dict:
        """
        Place a limit order.
        side: 'buy' or 'sell'
        Returns the order response dict.
        """
        payload = {
            "pair": pair,
            "side": side,
            "quantity": str(quantity),
            "price": str(price),
        }
        if custom_id:
            payload["custom_id"] = custom_id
        return self._request("POST", "/order", payload=payload)

    def place_market_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        custom_id: Optional[str] = None,
    ) -> Dict:
        """
        Place an instant (market) order.
        side: 'buy' or 'sell'
        """
        payload = {
            "pair": pair,
            "side": side,
            "quantity": str(quantity),
        }
        if custom_id:
            payload["custom_id"] = custom_id
        return self._request("POST", "/instant_order", payload=payload)

    def cancel_order(self, custom_id: str) -> Dict:
        """Cancel an open order by its custom_id."""
        return self._request(
            "PUT", f"/order/{custom_id}/cancel", params={"is_client_id": "true"}
        )

    def get_order(self, custom_id: str) -> Dict:
        """Fetch order details by custom_id."""
        return self._request("GET", "/order", params={"custom_id": custom_id, "is_client_id": "true"})

    def get_active_orders(self) -> Dict:
        """List all open orders."""
        return self._request("GET", "/order/active")

    def get_transactions(self, page: int = 1, per_page: int = 100) -> Dict:
        """List transaction history (trades, deposits, withdrawals)."""
        return self._request(
            "GET", "/transaction", params={"page": page, "per_page": per_page}
        )
