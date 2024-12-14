def read_int(statement: str = '') -> int:
  while True:
    try:
      return int(input(statement))
    except ValueError:
      print("Invalid input, try again")