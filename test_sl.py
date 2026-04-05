import traceback
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py").run(timeout=60)
    if at.exception:
        print("EXCEPTION CAUGHT BY APPTEST:")
        print(at.exception[0])
    else:
        print("NO EXCEPTION! IT RENDERED PERFECTLY.")
except Exception as e:
    print("FAILED TO RUN APPTEST:")
    traceback.print_exc()
