import pandas as pd
from tabulate import tabulate
import sys
import numpy as np


def get_col_from_aggr(col):
    for idx, char in enumerate(col):
        if char == '(':
            i = idx + 1
            s = ''
            while col[i] != ')':
                s += col[i]
                i += 1
            break
    return s


class QueryEngin:

    def __init__(self, base_path, test=False):
        self.tables = []  # List of tables
        self.tableCol = {}  # Map of table columns
        self.data = {}  # Table wise Data
        self.base_path = base_path
        self.operations = ["<", "<=", ">", ">=", '=']
        self.aggr = ["min", "max", "sum", "avg", "count"]
        self.query = {}
        self.results = None
        self.keywords = ['select', 'from', 'where', 'group', 'order', 'by', 'distinct']
        self.query_tokenized = None
        self.test = test

    def initialize(self):
        self.load_db_meta_data()
        self.load_data()
        pass

    def load_db_meta_data(self):
        db_meta_data = open(self.base_path + "/metadata.txt", 'r')
        lines = db_meta_data.readlines()
        # print(lines)
        cleaned = [line.strip() for line in lines]
        # print(cleaned)
        idx = 0
        while idx < len(cleaned):
            if cleaned[idx] == '<begin_table>':  # loop till <end_table>
                idx += 1
                table_name = cleaned[idx]
                self.tables.append(table_name)
                idx += 1
                cols = []
                while cleaned[idx] != '<end_table>':
                    cols.append(cleaned[idx])
                    idx += 1
                self.tableCol[table_name] = cols
            idx += 1

    def load_data(self):

        for table in self.tables:
            self.data[table] = pd.read_csv(self.base_path + '/' + table + '.csv'
                                           , header=None, names=self.tableCol[table])
            # print(tabulate(self.data[table], headers='keys', tablefmt='psql'))

    def join(self, tables):  # Full Cartesian product of multiple table
        cum = None
        curr = None
        for table in tables:

            if cum is None:
                cum = self.data[table]
                continue
            else:
                curr = self.data[table]
            cum = pd.merge(cum, curr, how='cross')
        return cum

    def error_check(self):
        pass

    def run(self, query):

        # do error check

        query = query.strip(";")
        self.deconstruct_query(query.split())
        self.apply_query()

    def deconstruct_query(self, query):

        self.query_preprocessing(query)
        self.get_distinct()
        self.get_columns()
        self.get_condition()
        self.get_working_table()
        self.get_aggregate_function()
        self.get_order_by()
        self.get_group_by()

        if self.test:
            print("######## Toknized Query #########")
            print(self.query_tokenized)

    def query_preprocessing(self, query):

        tokenize = []
        for token in query:
            if token.lower() in self.keywords:
                tokenize.append(token.lower())
            else:
                tokenize.append(token.strip(','))
        self.query_tokenized = tokenize

    def apply_query(self):

        self.apply_conditions()  # Where
        if self.query.get('aggr') is None and self.query.get('group') is None:
            self.apply_projection()
            self.apply_order_by()
            self.apply_distinct()
            self.print_table()
            if self.test:
                print("\n######### Results ###########\n")
                print(tabulate(self.results, headers='keys', tablefmt='psql'))
        elif self.query.get('group') is None:
            self.apply_aggr()
        elif self.query.get('group') is not None:
            self.apply_group_by()
            self.apply_distinct()
            self.apply_order_by()
            self.print_table()
            if self.test:
                print("\n######### Results ###########\n")
                print(tabulate(self.results, headers='keys', tablefmt='psql'))

    def apply_conditions(self):
        where = self.query['where']
        if where[0]:
            condition = where[1]

            op = None  # operation
            for i, condition_op in enumerate(condition):
                if i % 2 == 0:

                    lf = condition_op[0]
                    rt = condition_op[2]
                    operation = condition_op[1]
                    temp = None
                    if operation == '<':
                        temp = self.results[self.results[lf] < int(rt)]
                    elif operation == '>':
                        temp = self.results[self.results[lf] > int(rt)]
                    elif operation == '>=':
                        temp = self.results[self.results[lf] >= int(rt)]
                    elif operation == '<=':
                        temp = self.results[self.results[lf] <= int(rt)]
                    elif operation == '=':
                        temp = self.results[self.results[lf] == int(rt)]
                    if i == 0:
                        self.results = temp
                    else:
                        if op == "AND":
                            self.results = pd.merge(temp, self.results, how='inner')
                        elif op == "OR":
                            self.results = pd.merge(temp, self.results, how='outer')
                else:
                    op = condition_op
        else:
            if self.test:
                print("No Where condition")

    def get_condition(self):
        """
        [[condition], op ,[condition]]
        """
        conditions = []
        flag = False
        for i, token in enumerate(self.query_tokenized):
            if token == 'where':
                flag = True
                idx = i + 1

                while idx < len(self.query_tokenized):
                    if self.query_tokenized[idx] == 'group' or self.query_tokenized[idx] == 'order':
                        break
                    lr = None
                    op = None
                    if self.query_tokenized[idx].find(self.operations[1]) != -1:  # <=
                        lr = self.query_tokenized[idx].split(self.operations[1])
                        op = self.operations[1]
                        # print(self.operations[1], lr)
                    elif self.query_tokenized[idx].find(self.operations[3]) != -1:
                        lr = self.query_tokenized[idx].split(self.operations[3])
                        op = self.operations[3]
                        # print(self.operations[3], lr)
                    elif self.query_tokenized[idx].find(self.operations[0]) != -1:
                        lr = self.query_tokenized[idx].split(self.operations[0])
                        op = self.operations[0]
                        # print(self.operations[0],lr)
                    elif self.query_tokenized[idx].find(self.operations[2]) != -1:
                        lr = self.query_tokenized[idx].split(self.operations[2])
                        op = self.operations[2]
                        # print(self.operations[2], lr)
                    elif self.query_tokenized[idx].find(self.operations[4]) != -1:
                        lr = self.query_tokenized[idx].split(self.operations[4])
                        op = self.operations[4]
                    elif self.query_tokenized[idx].find('AND') != -1:  # Change according to query_tokenize
                        op = self.query_tokenized[idx]
                    elif self.query_tokenized[idx].find('OR') != -1:
                        op = self.query_tokenized[idx]
                    elif self.query_tokenized[idx].find('NOT') != -1:
                        op = self.query_tokenized[idx]
                    if lr is None:
                        conditions.append(op)
                    else:
                        conditions.append([lr[0], op, lr[1]])
                    idx += 1
        self.query['where'] = [flag, conditions]

    def get_working_table(self):
        tables_in_query = []
        for table in self.tables:
            if self.query_tokenized.count(table) != 0:
                tables_in_query.append(table)
        if len(tables_in_query) > 1:
            self.results = self.join(tables_in_query)
        else:
            self.results = self.data[tables_in_query[0]]

    def get_distinct(self):
        self.query['distinct'] = False
        for token in self.query_tokenized:
            if token == 'distinct':
                self.query['distinct'] = True
                break

    def apply_distinct(self):
        if self.query['distinct']:
            self.results = self.results.drop_duplicates()

    def get_columns(self):
        cols = []
        ind = 1
        if self.query['distinct']:
            ind += 1
            #print(self.query_tokenized[ind])
        if self.query_tokenized[ind] != '*':
            for idx, token in enumerate(self.query_tokenized):
                if idx < ind:
                    continue
                if token == 'from':
                    break
                cols.append(token)
            self.query['select'] = [False, cols]
        else:
            self.query['select'] = [True, cols]

    def apply_projection(self):
        select = self.query['select']
        if self.test:
            print("Projected Columns", select)
        if not select[0]:
            self.results = self.results[select[1]]

    def get_aggregate_function(self):
        select = self.query['select'][1]
        flag = False
        for col in select:
            for aggr in self.aggr:
                if col.find(aggr + '(') != -1:
                    self.query["aggr"] = [aggr, get_col_from_aggr(col)]
                    flag = True
                    break
            if flag:
                break

    def get_group_by(self):  # Can have multiple column
        for i, token in enumerate(self.query_tokenized):
            if i + 1 >= len(self.query_tokenized):
                break
            if token == 'group' and self.query_tokenized[1 + i] == 'by':
                i += 2
                self.query['group'] = self.query_tokenized[i]
                break

    def get_order_by(self):

        for i, token in enumerate(self.query_tokenized):
            if i + 1 >= len(self.query_tokenized):
                break
            if token == 'order' and self.query_tokenized[1 + i] == 'by':
                i += 2
                col = []
                ord = []
                while i < len(self.query_tokenized):
                    col_name = self.query_tokenized[i]
                    i += 1
                    order = None
                    if i < len(self.query_tokenized):
                        order = self.query_tokenized[i]
                    if order == 'DESC':
                        i += 1
                        col.append(col_name)
                        ord.append(False)
                    else:  # the token will either be ASC or col
                        col.append(col_name)
                        ord.append(True)
                self.query['order'] = [col, ord]
                break

    def apply_aggr(self):

        aggr = self.query['aggr']
        print(aggr[0] + "(" + aggr[1] + ")")
        if aggr is not None:
            print(self.results[aggr[1]].agg(aggr[0]))

    def apply_order_by(self):
        if self.query.get('order') is not None:
            self.results = self.results.sort_values(by=self.query['order'][0], ascending=self.query['order'][1])

    def apply_group_by(self):
        temp = None
        if self.query['aggr'][1] != '*':
            temp = self.results.groupby(self.query['group'])[self.query['aggr'][1]]
        else:
            cols = self.results.columns
            not_group = None
            for col in cols:
                if self.query['group'] != col:
                    not_group = col
                    break
            temp = self.results.groupby(self.query['group'])[not_group]
        data = []

        for group_id, group_data in temp:
            result = None
            if self.query['aggr'][0] == 'max':
                result = np.max(group_data)
            elif self.query['aggr'][0] == 'min':
                result = np.min(group_data)
            elif self.query['aggr'][0] == 'count':
                result = group_data.size
            elif self.query['aggr'][0] == 'sum':
                result = np.sum(group_data)
            elif self.query['aggr'][0] == 'avg':
                self.results = np.average(group_data)
            data.append([group_id, result])

        self.results = pd.DataFrame(data, columns=[self.query['group'],
                                                   self.query['aggr'][0] + "(" + self.query['aggr'][1] + ")"])

    def print_table(self):
        table_col = "< "
        for col in self.results.columns:
            for key, value in self.tableCol.items():
                if col in value:
                    table_col += key + '.' + col + ", "
                    break
        table_col = table_col.strip(', ')
        table_col += " >"
        print(table_col)
        for i in range(len(self.results.index)):
            row = ""
            for val in self.results.iloc[i]:
                row += str(val) + ", "
            row = row.strip(", ")
            print(row)


ql = ["select * from table1 where A>5 AND C<=4000", "select A, B from table1 where A>5 OR C<=4000",
      "select max(A) from table1 where A>5 OR C<=4000", "select distinct(A) from table1 where A>5 OR C<=4000"
    , "SelEct A, B FROM table1 WHERE A>5 OR C<=4000 order by A", "Select * from table1, table2"
    , "Select * from table2;", "Select C, A from table1;", "Select max(D) from table2;"
      ]
demoql = [
    "select distinct A, B from table1, table2",
    "Select A, count(*) from table1, table2 where A>5 OR C<=4000 group by A Order by A", "select * from table1"]

if len(sys.argv) > 1:
    qe = QueryEngin("./files")
    qe.initialize()
    qe.run(sys.argv[1])
else:
    for q in demoql:
        qe = QueryEngin("./files", test=True)
        qe.initialize()
        qe.run(q)
# distinct df.drop_duplicates()
# First we execute from then where then select(Projection)or then order by
# count(*) needs to be handled
