// A web-only panel that embeds the Flask-hosted Plotly charts via iframe
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// Import dart:html only on web builds
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;
// ignore: avoid_web_libraries_in_flutter
import 'dart:ui_web' as ui_web;

class PlotlyIFramePanel extends StatefulWidget {
  final int reloadToken;
  const PlotlyIFramePanel({super.key, required this.reloadToken});

  @override
  State<PlotlyIFramePanel> createState() => _PlotlyIFramePanelState();
}

class _PlotlyIFramePanelState extends State<PlotlyIFramePanel> {
  static const String _viewType = 'plotly-iframe-view';
  bool _registered = false;
  html.IFrameElement? _iframe;
  bool _backendAvailable = false;
  bool _checkingBackend = true;

  @override
  void initState() {
    super.initState();
    if (kIsWeb) {
      _checkBackendStatus();
      _registerViewFactoryOnce();
    }
  }

  Future<void> _checkBackendStatus() async {
    setState(() {
      _checkingBackend = true;
    });
    
    try {
      final response = await http.get(Uri.parse('/charts')).timeout(
        const Duration(seconds: 2),
      );
      setState(() {
        _backendAvailable = response.statusCode == 200;
        _checkingBackend = false;
      });
    } catch (e) {
      setState(() {
        _backendAvailable = false;
        _checkingBackend = false;
      });
    }
  }

  void _registerViewFactoryOnce() {
    if (_registered) return;
    _iframe = html.IFrameElement()
      ..src = '/charts'
      ..style.border = '0'
      ..style.width = '100%'
      ..style.height = '100%'
      ..allow = 'fullscreen'
      ..setAttribute('scrolling', 'auto');

    ui_web.platformViewRegistry
        .registerViewFactory(_viewType, (int viewId) => _iframe!);
    _registered = true;
  }

  @override
  void didUpdateWidget(covariant PlotlyIFramePanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (kIsWeb && oldWidget.reloadToken != widget.reloadToken) {
      _checkBackendStatus();
      final ts = DateTime.now().millisecondsSinceEpoch;
      _iframe?.src = '/charts?ts=$ts';
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb) {
      return const SizedBox.shrink();
    }
    
    if (_checkingBackend) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }
    
    if (!_backendAvailable) {
      return Center(
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: Colors.orange[50],
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: Colors.orange[200]!,
              width: 1,
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.cloud_off,
                color: Colors.orange[700],
                size: 48,
              ),
              const SizedBox(height: 16),
              Text(
                'Charts Unavailable',
                style: TextStyle(
                  color: Colors.orange[900],
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'The backend server is not running.\nPlease start the Flask server to view charts.',
                style: TextStyle(
                  color: Colors.orange[800],
                  fontSize: 14,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: _checkBackendStatus,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange[700],
                  foregroundColor: Colors.white,
                ),
              ),
            ],
          ),
        ),
      );
    }
    
    return const HtmlElementView(viewType: _viewType);
  }
}
