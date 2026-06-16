from workflow import run_workflow

while True:

    query = input("> ")

    if query.lower() == "exit":
        break

    answer = run_workflow(query)

    print(answer)
