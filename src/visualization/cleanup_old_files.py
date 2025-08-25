#!/usr/bin/env python3
"""
Cleanup script for visualization files.
Removes old timestamped files and keeps only the latest ones.
"""

import os
import glob
import re
from datetime import datetime

def cleanup_visualization_files(visualizations_dir="visualizations"):
    """Remove old timestamped files, keep only latest ones."""
    
    print("🧹 Cleaning up old visualization files...")
    
    # Get all files in the visualizations directory
    all_files = glob.glob(f"{visualizations_dir}/*")
    
    # Separate latest files from timestamped files
    latest_files = [f for f in all_files if "latest" in f]
    timestamped_files = [f for f in all_files if "latest" not in f and os.path.isfile(f)]
    
    print(f"📁 Found {len(latest_files)} latest files")
    print(f"📁 Found {len(timestamped_files)} timestamped files")
    
    # Remove timestamped files
    removed_count = 0
    for file_path in timestamped_files:
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"🗑️  Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️  Could not remove {file_path}: {e}")
    
    print(f"\n✅ Visualization cleanup completed!")
    print(f"🗑️  Removed {removed_count} old files")
    print(f"📁 Kept {len(latest_files)} latest files")
    
    # Show remaining files
    remaining_files = glob.glob(f"{visualizations_dir}/*")
    print(f"\n📊 Remaining visualization files:")
    for file_path in sorted(remaining_files):
        print(f"  • {os.path.basename(file_path)}")

def cleanup_data_files(data_dir="data/raw"):
    """Remove old data files, keep only the most recent one."""
    
    print("\n🧹 Cleaning up old data files...")
    
    # Get all synthetic data files
    data_files = glob.glob(f"{data_dir}/agetch_synthetic_data_*.csv")
    
    if len(data_files) <= 1:
        print("📁 Only one or no data files found - nothing to clean up")
        return
    
    # Sort by modification time to find the most recent
    data_files.sort(key=os.path.getmtime, reverse=True)
    most_recent_file = data_files[0]
    files_to_remove = data_files[1:]  # All except the most recent
    
    print(f"📁 Found {len(data_files)} data files")
    print(f"📁 Keeping most recent: {os.path.basename(most_recent_file)}")
    print(f"📁 Removing {len(files_to_remove)} old files")
    
    # Remove old files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"🗑️  Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️  Could not remove {file_path}: {e}")
    
    print(f"\n✅ Data cleanup completed!")
    print(f"🗑️  Removed {removed_count} old data files")
    print(f"📁 Kept: {os.path.basename(most_recent_file)}")
    
    # Show remaining data files
    remaining_files = glob.glob(f"{data_dir}/agetch_synthetic_data_*.csv")
    print(f"\n📊 Remaining data files:")
    for file_path in sorted(remaining_files):
        print(f"  • {os.path.basename(file_path)}")

def run_complete_cleanup():
    """Run complete cleanup of both visualizations and data files."""
    print("🧹 COMPLETE CLEANUP - Removing Old Files")
    print("=" * 50)
    
    # Clean up visualization files
    cleanup_visualization_files()
    
    # Clean up data files
    cleanup_data_files()
    
    print("\n" + "=" * 50)
    print("✅ COMPLETE CLEANUP FINISHED!")
    print("=" * 50)
    print("📁 Now you have:")
    print("  • Only the most recent data file")
    print("  • Only the latest visualization files")
    print("  • Clean, organized directories")

if __name__ == "__main__":
    run_complete_cleanup()
