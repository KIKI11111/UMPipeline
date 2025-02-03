from enum import Enum

class test(Enum):
    USERID = (1, 'userid', 'table')

    def __init__(self, id, username, table):
        self.id = id
        self.username = username
        self.table = table

print(test.USERID)
print(test.USERID.id)