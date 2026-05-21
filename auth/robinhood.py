import pyotp
import robin_stocks.robinhood as rh
from config import RH_USERNAME, RH_PASSWORD, RH_MFA_SECRET
from rich.console import Console
from rich.prompt import Prompt

console = Console()
_logged_in = False


def login(force: bool = False) -> bool:
    global _logged_in
    if _logged_in and not force:
        return True

    username = RH_USERNAME or Prompt.ask("[bold cyan]Robinhood email")
    password = RH_PASSWORD or Prompt.ask("[bold cyan]Robinhood password", password=True)

    mfa_code = None
    if RH_MFA_SECRET:
        mfa_code = pyotp.TOTP(RH_MFA_SECRET).now()

    try:
        result = rh.login(
            username=username,
            password=password,
            mfa_code=mfa_code,
            store_session=True,
            by_sms=not bool(RH_MFA_SECRET),
        )
        if result:
            _logged_in = True
            console.print("[bold green]✓ Authenticated with Robinhood[/]")
            return True
    except Exception as e:
        console.print(f"[bold red]Auth failed: {e}[/]")

    return False


def logout():
    global _logged_in
    rh.logout()
    _logged_in = False
