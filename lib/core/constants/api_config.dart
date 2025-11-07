class ApiConfig {
  ApiConfig._();

  static const String defaultBaseUrl =
      'https://qd4u7ojkn92tz2-8000.proxy.runpod.net';

  static const String baseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: defaultBaseUrl);

  static String resolve(String path) {
    if (path.isEmpty || path == '/') {
      return baseUrl;
    }
    return path.startsWith('/') ? '$baseUrl$path' : '$baseUrl/$path';
  }
}
