from dotenv import load_dotenv

from graph.graph import app


if __name__ == '__main__':
    # print(app.invoke(
    #     input={'question': 'what is agent memory?'}
    # ))
    print(app.invoke(
        input={'question': 'what to make pizza?'}
    ))