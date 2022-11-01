import sqlite3

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class BagFileParser():

    def __init__(self, bag_file):
        try:
            self.conn = sqlite3.connect(bag_file)
        except Exception as e:
            print('Could not connect: ', e)
            raise Exception('could not connect')

        self.cursor = self.conn.cursor()

        ## create a message (id, topic, type) map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

    # Return messages as list of tuples [(timestamp0, message0), (timestamp1, message1), ...]
    def get_bag_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]

    def get_bag_msgs_iter(self, topic_name):
        topic_id = self.topic_id[topic_name]
        result = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id))
        while True:
            res = result.fetchone()
            if res is not None:
                yield (res[0], deserialize_message(res[1], self.topic_msg_message[topic_name]))
            else:
                break
