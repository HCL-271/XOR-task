hello-py
===

Setup instructions:

1. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Install dependencies:
   ```
   pip3 install .
   ```

3. Run the agent:
   ```
   python3 main.py
   ```

4. Run the tests:
   ```
   pytest
   ```

5. All required changes belong in `task.py`

## See transcript outputs

By default, `verbose=True` in `main.py`. Set `verbose=False` to turn off transcript outputs.

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.