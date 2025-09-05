"""
Command-line interface for FloatChat.

This module provides CLI commands for various FloatChat operations including
data processing, server management, and development utilities.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from floatchat.core.config import settings
from floatchat.core.logging import get_logger

# Initialize CLI app and console
app = typer.Typer(
    name="floatchat",
    help="FloatChat: AI-Powered Conversational Interface for ARGO Ocean Data",
    rich_markup_mode="rich"
)
console = Console()
logger = get_logger(__name__)


@app.command()
def version():
    """Show FloatChat version information."""
    from floatchat import __version__, __title__, __description__
    
    table = Table(title="FloatChat Version Information", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Application", __title__)
    table.add_row("Version", __version__)
    table.add_row("Description", __description__)
    table.add_row("Environment", settings.environment)
    table.add_row("Debug Mode", str(settings.debug))
    
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(settings.api.api_host, help="Host to bind to"),
    port: int = typer.Option(settings.api.api_port, help="Port to bind to"),
    workers: int = typer.Option(settings.api.api_workers, help="Number of workers"),
    reload: bool = typer.Option(settings.api.api_reload, help="Enable auto-reload"),
):
    """Start the FloatChat API server."""
    rprint(f"[bold green]Starting FloatChat API server...[/bold green]")
    rprint(f"[cyan]Environment:[/cyan] {settings.environment}")
    rprint(f"[cyan]Host:[/cyan] {host}")
    rprint(f"[cyan]Port:[/cyan] {port}")
    rprint(f"[cyan]Workers:[/cyan] {workers}")
    
    uvicorn.run(
        "floatchat.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload and settings.is_development,
        log_config=None,
        access_log=False,
    )


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8501, help="Port to bind to"),
):
    """Start the FloatChat Streamlit dashboard."""
    import subprocess
    import sys
    
    rprint(f"[bold green]Starting FloatChat dashboard...[/bold green]")
    rprint(f"[cyan]Host:[/cyan] {host}")
    rprint(f"[cyan]Port:[/cyan] {port}")
    rprint(f"[yellow]Dashboard will be available at:[/yellow] http://{host}:{port}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/floatchat/presentation/dashboard.py",
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        rprint("[yellow]Dashboard stopped.[/yellow]")


@app.command()
def health():
    """Check FloatChat system health."""
    import httpx
    
    rprint("[bold blue]Checking FloatChat system health...[/bold blue]")
    
    try:
        with httpx.Client() as client:
            response = client.get(f"http://{settings.api.api_host}:{settings.api.api_port}/health")
            health_data = response.json()
            
            # Display overall status
            status = health_data.get("status", "unknown")
            status_color = {
                "healthy": "green",
                "degraded": "yellow", 
                "unhealthy": "red"
            }.get(status, "white")
            
            rprint(f"[bold {status_color}]Overall Status: {status.upper()}[/bold {status_color}]")
            rprint(f"[cyan]Version:[/cyan] {health_data.get('version', 'unknown')}")
            rprint(f"[cyan]Environment:[/cyan] {health_data.get('environment', 'unknown')}")
            rprint(f"[cyan]Uptime:[/cyan] {health_data.get('uptime_seconds', 0):.2f} seconds")
            
            # Display component status
            checks = health_data.get("checks", {})
            if checks:
                rprint("\n[bold]Component Status:[/bold]")
                
                table = Table()
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="bold")
                table.add_column("Response Time", style="yellow")
                table.add_column("Message", style="dim")
                
                for component, check in checks.items():
                    comp_status = check.get("status", "unknown")
                    comp_color = {
                        "healthy": "green",
                        "degraded": "yellow",
                        "unhealthy": "red"
                    }.get(comp_status, "white")
                    
                    response_time = check.get("response_time_ms", 0)
                    message = check.get("message", "")
                    
                    table.add_row(
                        component.replace("_", " ").title(),
                        f"[{comp_color}]{comp_status.upper()}[/{comp_color}]",
                        f"{response_time:.2f}ms" if response_time else "N/A",
                        message[:50] + "..." if len(message) > 50 else message
                    )
                
                console.print(table)
                
    except httpx.ConnectError:
        rprint("[red]Error: Cannot connect to FloatChat API server.[/red]")
        rprint("[yellow]Make sure the server is running with 'floatchat serve'[/yellow]")
    except Exception as e:
        rprint(f"[red]Error checking health: {e}[/red]")


# Data processing commands group
data_app = typer.Typer(help="Data processing commands")
app.add_typer(data_app, name="data")


@data_app.command("download")
def download_argo_data(
    limit: int = typer.Option(10, help="Number of files to download"),
    region: Optional[str] = typer.Option(None, help="Geographic region filter"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
):
    """Download ARGO NetCDF files for processing."""
    rprint(f"[bold blue]Downloading ARGO data...[/bold blue]")
    rprint(f"[cyan]Limit:[/cyan] {limit} files")
    rprint(f"[cyan]Region:[/cyan] {region or 'Global'}")
    rprint(f"[cyan]Output:[/cyan] {output_dir or settings.data.argo_data_path}")
    
    # This will be implemented in Phase 1.3
    rprint("[yellow]Data download functionality will be implemented in Phase 1.3[/yellow]")


@data_app.command("process")
def process_netcdf_files(
    input_dir: Optional[Path] = typer.Option(None, help="Input directory"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
    concurrent: int = typer.Option(5, help="Concurrent processing limit"),
):
    """Process ARGO NetCDF files into database."""
    rprint(f"[bold blue]Processing ARGO NetCDF files...[/bold blue]")
    rprint(f"[cyan]Input:[/cyan] {input_dir or settings.data.argo_data_path}")
    rprint(f"[cyan]Concurrent:[/cyan] {concurrent}")
    
    # This will be implemented in Phase 1.3
    rprint("[yellow]NetCDF processing functionality will be implemented in Phase 1.3[/yellow]")


@data_app.command("analyze")
def analyze_netcdf_structure(
    file_path: Path = typer.Argument(..., help="Path to NetCDF file to analyze")
):
    """Analyze ARGO NetCDF file structure."""
    rprint(f"[bold blue]Analyzing NetCDF file structure...[/bold blue]")
    rprint(f"[cyan]File:[/cyan] {file_path}")
    
    if not file_path.exists():
        rprint(f"[red]Error: File {file_path} does not exist[/red]")
        raise typer.Exit(1)
    
    # This will be implemented in the next step
    rprint("[yellow]NetCDF analysis functionality will be implemented next[/yellow]")


# Database commands group
db_app = typer.Typer(help="Database management commands")
app.add_typer(db_app, name="db")


@db_app.command("upgrade")
def db_upgrade():
    """Run database migrations."""
    rprint("[bold blue]Running database migrations...[/bold blue]")
    
    async def run_migration():
        try:
            from floatchat.infrastructure.database.service import db_service
            await db_service.initialize_database()
            rprint("[green]✓ Database migrations completed successfully[/green]")
        except Exception as e:
            rprint(f"[red]✗ Migration failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_migration())


@db_app.command("reset")
def db_reset(
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation prompt")
):
    """Reset database (development only)."""
    if settings.is_production:
        rprint("[red]Error: Database reset is not allowed in production![/red]")
        raise typer.Exit(1)
    
    if not confirm:
        confirm = typer.confirm("This will destroy all data. Are you sure?")
    
    if confirm:
        rprint("[bold red]Resetting database...[/bold red]")
        
        async def run_reset():
            try:
                from floatchat.infrastructure.database.service import db_service
                await db_service.reset_database()
                rprint("[green]✓ Database reset completed successfully[/green]")
            except Exception as e:
                rprint(f"[red]✗ Database reset failed: {e}[/red]")
                raise typer.Exit(1)
        
        asyncio.run(run_reset())
    else:
        rprint("[yellow]Database reset cancelled.[/yellow]")


@db_app.command("status")
def db_status():
    """Show database status and statistics."""
    rprint("[bold blue]Checking database status...[/bold blue]")
    
    async def check_status():
        try:
            from floatchat.infrastructure.database.service import db_service
            
            # Health check
            health = await db_service.health_check()
            
            if health["status"] == "healthy":
                rprint("[green]✓ Database is healthy[/green]")
                rprint(f"[cyan]Response Time:[/cyan] {health.get('response_time_ms', 0):.2f}ms")
                if "tables_found" in health:
                    rprint(f"[cyan]Tables Found:[/cyan] {health['tables_found']}")
                if "migrations_applied" in health:
                    rprint(f"[cyan]Migrations Applied:[/cyan] {health['migrations_applied']}")
            else:
                rprint(f"[red]✗ Database is {health['status']}[/red]")
                rprint(f"[red]Reason:[/red] {health['message']}")
                return
            
            # Get statistics
            stats = await db_service.get_database_statistics()
            
            if "error" not in stats:
                rprint("\n[bold]Database Statistics:[/bold]")
                rprint(f"[cyan]Total Floats:[/cyan] {stats.get('total_floats', 0)}")
                rprint(f"[cyan]Active Floats:[/cyan] {stats.get('active_floats', 0)}")
                rprint(f"[cyan]Total Profiles:[/cyan] {stats.get('total_profiles', 0)}")
                rprint(f"[cyan]Total Measurements:[/cyan] {stats.get('total_measurements', 0)}")
                
                if stats.get('latest_profile_date'):
                    rprint(f"[cyan]Latest Profile:[/cyan] {stats['latest_profile_date']}")
                if stats.get('earliest_profile_date'):
                    rprint(f"[cyan]Earliest Profile:[/cyan] {stats['earliest_profile_date']}")
            
        except Exception as e:
            rprint(f"[red]✗ Status check failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(check_status())


@db_app.command("optimize")
def db_optimize():
    """Optimize database performance (VACUUM ANALYZE)."""
    rprint("[bold blue]Optimizing database performance...[/bold blue]")
    
    async def run_optimize():
        try:
            from floatchat.infrastructure.database.service import db_service
            await db_service.vacuum_analyze()
            rprint("[green]✓ Database optimization completed successfully[/green]")
        except Exception as e:
            rprint(f"[red]✗ Database optimization failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_optimize())


# AI commands group
ai_app = typer.Typer(help="AI and ML commands")
app.add_typer(ai_app, name="ai")


@ai_app.command("embeddings")
def generate_embeddings(
    rebuild: bool = typer.Option(False, help="Rebuild all embeddings"),
    batch_size: int = typer.Option(100, help="Batch size for processing"),
):
    """Generate embeddings for ARGO data."""
    rprint(f"[bold blue]Generating embeddings...[/bold blue]")
    rprint(f"[cyan]Rebuild:[/cyan] {rebuild}")
    rprint(f"[cyan]Batch size:[/cyan] {batch_size}")
    
    # This will be implemented in Phase 2
    rprint("[yellow]Embedding generation functionality will be implemented in Phase 2[/yellow]")


@ai_app.command("index")
def build_vector_index(
    index_type: str = typer.Option("HNSW", help="FAISS index type"),
    rebuild: bool = typer.Option(False, help="Rebuild existing index"),
):
    """Build FAISS vector search index."""
    rprint(f"[bold blue]Building vector search index...[/bold blue]")
    rprint(f"[cyan]Index type:[/cyan] {index_type}")
    rprint(f"[cyan]Rebuild:[/cyan] {rebuild}")
    
    # This will be implemented in Phase 2
    rprint("[yellow]Vector index building functionality will be implemented in Phase 2[/yellow]")


# Development commands group
dev_app = typer.Typer(help="Development utilities")
app.add_typer(dev_app, name="dev")


@dev_app.command("shell")
def interactive_shell():
    """Start interactive Python shell with FloatChat context."""
    import IPython
    from floatchat.core.config import settings
    from floatchat.core.logging import get_logger
    
    rprint("[bold green]Starting FloatChat interactive shell...[/bold green]")
    
    # Prepare shell context
    context = {
        'settings': settings,
        'logger': get_logger('shell'),
    }
    
    rprint("[dim]Available objects: settings, logger[/dim]")
    
    IPython.start_ipython(argv=[], user_ns=context)


@dev_app.command("test")
def run_tests(
    coverage: bool = typer.Option(False, help="Run with coverage"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    pattern: Optional[str] = typer.Option(None, help="Test pattern filter"),
):
    """Run the test suite."""
    import subprocess
    import sys
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if coverage:
        cmd.extend(["--cov=floatchat", "--cov-report=html", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    if pattern:
        cmd.extend(["-k", pattern])
    
    rprint(f"[bold blue]Running tests...[/bold blue]")
    rprint(f"[cyan]Command:[/cyan] {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        rprint("[yellow]Tests interrupted.[/yellow]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()