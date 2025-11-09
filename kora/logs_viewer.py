#!/usr/bin/env python3
"""
KORA Database Logs Viewer CLI
View and analyze KORA logs from the database
"""
import asyncio
import argparse
from typing import Optional
from datetime import datetime, timedelta
from tabulate import tabulate

from kora.db_logger import get_db, get_database_url


async def view_query_logs(limit: int = 20, user_id: Optional[str] = None):
    """View recent query logs"""
    db = await get_db()
    
    where_clause = {}
    if user_id:
        where_clause["userId"] = user_id
    
    logs = await db.querylog.find_many(
        where=where_clause,
        order={"timestamp": "desc"},
        take=limit
    )
    
    if not logs:
        print("No query logs found")
        return
    
    table_data = []
    for log in logs:
        table_data.append([
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            log.username or log.userId[:8] if log.userId else "N/A",
            log.query[:50] + "..." if len(log.query) > 50 else log.query,
            log.model,
            "✅" if log.success else "❌",
            f"{log.duration:.2f}s" if log.duration else "N/A"
        ])
    
    print("\n📊 Recent Query Logs\n")
    print(tabulate(
        table_data,
        headers=["Timestamp", "User", "Query", "Model", "Status", "Duration"],
        tablefmt="grid"
    ))
    print(f"\nTotal: {len(logs)} queries")


async def view_auth_logs(limit: int = 20):
    """View recent authentication logs"""
    db = await get_db()
    
    logs = await db.authlog.find_many(
        order={"timestamp": "desc"},
        take=limit
    )
    
    if not logs:
        print("No auth logs found")
        return
    
    table_data = []
    for log in logs:
        table_data.append([
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            log.action,
            log.username or log.userId[:8] if log.userId else "N/A",
            "✅" if log.success else "❌",
            log.reason or "-",
            log.ipAddress or "N/A"
        ])
    
    print("\n🔐 Recent Authentication Logs\n")
    print(tabulate(
        table_data,
        headers=["Timestamp", "Action", "User", "Status", "Reason", "IP"],
        tablefmt="grid"
    ))
    print(f"\nTotal: {len(logs)} events")


async def view_system_logs(limit: int = 20, level: Optional[str] = None):
    """View recent system logs"""
    db = await get_db()
    
    where_clause = {}
    if level:
        where_clause["level"] = level.upper()
    
    logs = await db.systemlog.find_many(
        where=where_clause,
        order={"timestamp": "desc"},
        take=limit
    )
    
    if not logs:
        print("No system logs found")
        return
    
    table_data = []
    for log in logs:
        level_icon = {
            "DEBUG": "🐛",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🔥"
        }.get(log.level, "📝")
        
        table_data.append([
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"{level_icon} {log.level}",
            log.component,
            log.message[:60] + "..." if len(log.message) > 60 else log.message
        ])
    
    print("\n🔧 Recent System Logs\n")
    print(tabulate(
        table_data,
        headers=["Timestamp", "Level", "Component", "Message"],
        tablefmt="grid"
    ))
    print(f"\nTotal: {len(logs)} events")


async def view_stats():
    """View usage statistics"""
    db = await get_db()
    
    # Query stats
    total_queries = await db.querylog.count()
    successful_queries = await db.querylog.count(where={"success": True})
    failed_queries = total_queries - successful_queries
    
    # User stats
    unique_query_users = len(await db.querylog.find_many(distinct=["userId"]))
    unique_active_users = len(await db.activitylog.find_many(distinct=["userId"]))
    
    # Auth stats
    total_auth_events = await db.authlog.count()
    failed_auth = await db.authlog.count(where={"success": False})
    
    # System stats
    error_logs = await db.systemlog.count(where={"level": "ERROR"})
    warning_logs = await db.systemlog.count(where={"level": "WARNING"})
    
    # Recent activity (last 24h)
    yesterday = datetime.now() - timedelta(days=1)
    recent_queries = await db.querylog.count(
        where={"timestamp": {"gte": yesterday}}
    )
    
    print("\n" + "=" * 60)
    print("  📊 KORA Usage Statistics")
    print("=" * 60 + "\n")
    
    print("Queries:")
    print(f"  • Total queries: {total_queries}")
    print(f"  • Successful: {successful_queries} ({successful_queries/total_queries*100:.1f}%)" if total_queries > 0 else "  • Successful: 0")
    print(f"  • Failed: {failed_queries}")
    print(f"  • Last 24h: {recent_queries}")
    
    print("\nUsers:")
    print(f"  • Unique query users: {unique_query_users}")
    print(f"  • Unique active users: {unique_active_users}")
    
    print("\nAuthentication:")
    print(f"  • Total auth events: {total_auth_events}")
    print(f"  • Failed attempts: {failed_auth}")
    
    print("\nSystem:")
    print(f"  • Error logs: {error_logs}")
    print(f"  • Warning logs: {warning_logs}")
    
    print("\n" + "=" * 60 + "\n")


async def export_logs(output_file: str, log_type: str = "all"):
    """Export logs to CSV"""
    import csv
    
    db = await get_db()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if log_type in ["all", "query"]:
            logs = await db.querylog.find_many(order={"timestamp": "desc"})
            writer.writerow([
                "Timestamp", "User ID", "Username", "Query", "Response",
                "Model", "Temperature", "Top K", "Success", "Duration"
            ])
            for log in logs:
                writer.writerow([
                    log.timestamp, log.userId, log.username, log.query,
                    log.response, log.model, log.temperature, log.topK,
                    log.success, log.duration
                ])
    
    print(f"✅ Logs exported to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="KORA Database Logs Viewer")
    parser.add_argument(
        "command",
        choices=["queries", "auth", "system", "stats", "export"],
        help="Command to execute"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of records to display (default: 20)"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Filter by user ID"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Filter system logs by level"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for export command"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "query", "auth", "system"],
        help="Log type to export"
    )
    
    args = parser.parse_args()
    
    print(f"\n🔌 Connecting to database: {get_database_url()}\n")
    
    try:
        if args.command == "queries":
            await view_query_logs(args.limit, args.user)
        elif args.command == "auth":
            await view_auth_logs(args.limit)
        elif args.command == "system":
            await view_system_logs(args.limit, args.level)
        elif args.command == "stats":
            await view_stats()
        elif args.command == "export":
            if not args.output:
                print("❌ Error: --output required for export command")
                return
            await export_logs(args.output, args.type)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
