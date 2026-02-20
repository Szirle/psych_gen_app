class ApiConfig {
  ApiConfig._();

  // Keep relative by default so containerized deployments can inject host/port
  // at runtime (same-origin requests like `/images`).
  static const String defaultBaseUrl = '';

  static const String baseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: defaultBaseUrl);

  static String resolve(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';

    if (baseUrl.isEmpty) {
      return normalizedPath;
    }

    if (path.isEmpty || path == '/') {
      return baseUrl;
    }

    if (baseUrl.endsWith('/')) {
      return '$baseUrl${normalizedPath.substring(1)}';
    }

    return '$baseUrl$normalizedPath';
  }
}
