import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension_name.dart';
import 'package:psych_gen_app/features/face_generation/presentation/bloc/filters_bloc.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/distribution_range_selector.dart';

String _humanizeRaw(String s) {
  if (s.isEmpty) return s;
  final buffer = StringBuffer();
  for (int i = 0; i < s.length; i++) {
    final String c = s[i];
    final bool isUpper = c.toUpperCase() == c && c.toLowerCase() != c;
    final bool prevIsLower = i > 0 && s[i - 1].toLowerCase() == s[i - 1];
    if (i > 0 && isUpper && prevIsLower) buffer.write(' ');
    buffer.write(c);
  }
  return buffer
      .toString()
      .split(' ')
      .map((w) => w.isEmpty ? w : (w[0].toUpperCase() + w.substring(1)))
      .join(' ');
}

String _labelForEnum(ManipulatedDimensionName name) => _humanizeRaw(name.name);

class FiltersPanel extends StatelessWidget {
  final List<ManipulatedDimension> currentDims;
  const FiltersPanel({super.key, required this.currentDims});

  @override
  Widget build(BuildContext context) {
    return ExpansionTile(
      initiallyExpanded: false,
      maintainState: true,
      title: Text(
        'section.filters'.tr(),
        style: const TextStyle(
          fontFamily: 'WorkSans',
          fontWeight: FontWeight.bold,
        ),
      ),
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
          child: BlocBuilder<FiltersBloc, FiltersState>(
            builder: (context, state) {
              if (state is FiltersInitial) {
                context.read<FiltersBloc>().add(
                      LoadDistributionsEvent(
                        variables: ManipulatedDimensionName.values,
                      ),
                    );
              }
              if (state is FiltersLoading || state is FiltersInitial) {
                return const Padding(
                  padding: EdgeInsets.all(12.0),
                  child: LinearProgressIndicator(),
                );
              }
              if (state is FiltersError) {
                return Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Text(state.message,
                      style: const TextStyle(color: Colors.red)),
                );
              }
              final loaded = state as FiltersLoaded;
              final entries = ManipulatedDimensionName.values;
              return Column(
                children: entries.map((name) {
                  final dim = currentDims.firstWhere(
                    (d) => d.name == name,
                    orElse: () => ManipulatedDimension(
                        name: name, strength: 25.0, nLevels: 2),
                  );
                  final dist = loaded.distributions[name] ?? const <double>[];
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 6.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Padding(
                          padding: const EdgeInsets.only(bottom: 6.0),
                          child: Text(
                            _labelForEnum(name),
                            style: const TextStyle(
                              fontFamily: 'WorkSans',
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF2B3A55),
                            ),
                          ),
                        ),
                        DistributionRangeSelector(
                          dimension: dim,
                          accentColor: const Color(0xFF2B3A55),
                          values: dist,
                          currentStart:
                              loaded.appliedFilters[name]?.elementAt(0),
                          currentEnd: loaded.appliedFilters[name]?.elementAt(1),
                          onRangeChanged: (start, end) {
                            // Only update local filters; do not refetch distributions here
                            context.read<FiltersBloc>().add(
                                  UpdateFilterEvent(
                                    dimension: name,
                                    range: [start, end],
                                  ),
                                );
                          },
                        ),
                      ],
                    ),
                  );
                }).toList(),
              );
            },
          ),
        ),
      ],
    );
  }
}
