"""Describe the purpose of this script here..."""
import click
from codetiming import Timer
from humanfriendly import format_timespan


@click.command()
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main():
    pass
    

if __name__ == "__main__":
    main()
