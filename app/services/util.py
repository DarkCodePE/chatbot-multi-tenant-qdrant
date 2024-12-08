from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifica si la contraseña en texto plano coincide con el hash almacenado.

    :param plain_password: La contraseña en texto plano proporcionada por el usuario.
    :param hashed_password: El hash de la contraseña almacenado en la base de datos.
    :return: True si la contraseña coincide, False en caso contrario.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Genera un hash seguro para la contraseña proporcionada.

    :param password: La contraseña en texto plano a hashear.
    :return: El hash de la contraseña.
    """
    return pwd_context.hash(password)
