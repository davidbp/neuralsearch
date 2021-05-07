
from jina import Executor

class MyExecutor(Executor):

  def __init__(self, bar: int,**kwargs):
    super().__init__( **kwargs)
    self.bar = bar

  def foo(self, **kwargs):
    pass

y_literal = """
jtype: MyExecutor
with:
  bar: 123
metas:
  name: awesomeness
  description: my first awesome executor
requests:
  /random_work: foo
"""

ex = Executor.load_config(y_literal)
ex.save_config('y.yml')

ex_from_load = Executor.load_config('y.yml')
#ex2 = MyExecutor(bar=10, weight=20)
#ex.save_config('y_2.yml')
