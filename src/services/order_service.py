"""
Order Service — SQLite-backed order management.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from src.services.database import get_connection


@dataclass
class OrderItem:
    product_name: str
    sku: str
    quantity: int
    price: float


@dataclass
class Order:
    order_id: str
    customer_name: str
    email: str
    items: list[OrderItem]
    status: str  # confirmed, shipped, delivered, cancelled
    total: float
    created_at: str
    shipping_address: str
    tracking_number: Optional[str] = None


def _row_to_order(row) -> Order:
    items = [OrderItem(**i) for i in json.loads(row["items"])]
    return Order(
        order_id=row["order_id"],
        customer_name=row["customer_name"],
        email=row["email"],
        items=items,
        status=row["status"],
        total=row["total"],
        created_at=row["created_at"],
        shipping_address=row["shipping_address"],
        tracking_number=row["tracking_number"],
    )


class OrderService:
    """SQLite-backed order management."""

    def get_order(self, order_id: str) -> Optional[Order]:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id.upper(),)
            ).fetchone()
            return _row_to_order(row) if row else None
        finally:
            conn.close()

    def get_orders_by_email(self, email: str) -> list[Order]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM orders WHERE email = ? ORDER BY created_at DESC",
                (email.lower(),),
            ).fetchall()
            return [_row_to_order(r) for r in rows]
        finally:
            conn.close()

    def get_all_orders(self) -> list[Order]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC"
            ).fetchall()
            return [_row_to_order(r) for r in rows]
        finally:
            conn.close()

    def search_orders(self, query: str) -> list[Order]:
        """Search by order ID, customer name, or product name."""
        conn = get_connection()
        try:
            pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT DISTINCT o.* FROM orders o
                   WHERE o.order_id LIKE ?
                      OR o.customer_name LIKE ?
                      OR o.items LIKE ?
                   ORDER BY o.created_at DESC""",
                (pattern, pattern, pattern),
            ).fetchall()
            return [_row_to_order(r) for r in rows]
        finally:
            conn.close()

    def can_return(self, order_id: str) -> tuple[bool, str]:
        """Check if an order is eligible for return (within 14 days, delivered)."""
        order = self.get_order(order_id)
        if not order:
            return False, f"Order {order_id} not found."
        if order.status == "cancelled":
            return False, "Cancelled orders cannot be returned."
        if order.status != "delivered":
            return False, f"Order has not been delivered yet (status: {order.status})."
        created = datetime.fromisoformat(order.created_at)
        if datetime.now() - created > timedelta(days=14):
            return False, "The 14-day return window has expired."
        return True, "Order is eligible for return."


    
    def cancel_order(
        self, order_id: str, authenticated_email: str
    ) -> tuple[bool, str]:
        """Cancel a confirmed order belonging to the authenticated user."""
        order_id = order_id.upper()
        authenticated_email = authenticated_email.strip().lower()

        if not authenticated_email:
            return False, "Authenticated customer identity is required."

        conn = get_connection()

        try:
            cursor = conn.execute(
                """
                UPDATE orders
                SET status = 'cancelled'
                WHERE order_id = ?
                  AND LOWER(email) = ?
                  AND status = 'confirmed'
                """,
                (order_id, authenticated_email),
            )

            if cursor.rowcount == 1:
                conn.commit()
                return True, f"Order {order_id} cancelled successfully."

            row = conn.execute(
                "SELECT email, status FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()

            if row is None:
                return False, f"Order {order_id} not found."

            if row["email"].lower() != authenticated_email:
                return False, "Not authorized to cancel this order."

            return False, (
                f"Cannot cancel order {order_id}. "
                f"Current status: {row['status']}."
            )

        finally:
            conn.close()
