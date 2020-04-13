


import json
import tornado.web
import tornado.ioloop
from tornado.options import define, options

class MBaseService(tornado.web.RequestHandler):

    def get(self):
        self.post()

    def get_json_argument(self, name, default=None):
        args = json_decode(self.request.body)
        name = to_unicode(name)
        if name in args:
            return args[name]
        elif default is not None:
            return default
        else:
            raise tornado.web.MissingArgumentError(name)

    def post(self):
        # print(self.)
        q = self.get_argument('q')
        b = self.get_argument('b')
        print(q, b)
        self.write(json.dumps({"q": q, "b": b}, ensure_ascii=False))


def make_app():
    app = tornado.web.Application([(r'/index', MBaseService)])
    return app


if __name__ == '__main__':
    app = make_app()
    app.listen(7878)
    tornado.ioloop.IOLoop.instance().start()
