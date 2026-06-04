import string


def generate_bytes_for_transmission(filename, content: bytes):
    length_of_file = len(content)

    data = bytearray()
    data.extend(filename.encode())
    data.extend(b"\0")
    data.extend(str(length_of_file).encode())
    data.extend(b"\0")
    data.extend(content)

    return data


def decode_received_file(data: bytes):
    filename, num_bytes, padded_content = data.split(b"\0", 2)
    length = int(num_bytes)

    sanitized_filename = "".join(
        filter(
            lambda char: char in (string.digits + string.ascii_letters + ".-_"),
            filename.decode("ascii", "ignore"),
        )
    )
    filename = f"output/file_{sanitized_filename}"

    print(f"Read file length {length} bytes")
    print(f"Writing to file: {filename}")

    return filename, padded_content[:length]
