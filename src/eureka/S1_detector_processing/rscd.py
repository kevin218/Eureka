import logging

from stdatamodels.jwst import datamodels

from jwst.rscd import rscd_sub
from jwst.rscd.rscd_step import RscdStep


log = logging.getLogger(__name__)

__all__ = ["Eureka_RscdStep"]


class Eureka_RscdStep(RscdStep):
    """Run the JWST RSCD correction with optional group-count overrides.

    This step extends :class:`jwst.rscd.rscd_step.RscdStep` by allowing
    Eureka! control files to override the number of initial groups flagged in
    the first integration and in subsequent integrations independently. An
    override of ``None`` retains the corresponding value from the CRDS RSCD
    reference file.

    Attributes
    ----------
    group_skip1 : int or None
        Number of initial groups to flag in the first integration. If None,
        use ``group_skip1`` from the CRDS RSCD reference file.
    group_skip : int or None
        Number of initial groups to flag in the second and subsequent
        integrations. If None, use ``group_skip`` from the CRDS RSCD
        reference file.

    Notes
    -----
    The selected group counts are passed to the upstream JWST RSCD correction.
    Its protections for short ramps and rapidly saturating pixels therefore
    remain active and may reduce the number of groups flagged for some data.
    """

    spec = """
        group_skip1 = integer(default=None) # Initial groups to flag in integration 1
        group_skip = integer(default=None)  # Initial groups to flag in integrations 2+
    """  # noqa: E501

    def process(self, step_input):
        """Flag initial MIRI groups using CRDS or user-supplied counts.

        Parameters
        ----------
        step_input : str or stdatamodels.jwst.datamodels.RampModel
            Input MIRI ramp model or path to a ramp-model file.

        Returns
        -------
        result : stdatamodels.jwst.datamodels.RampModel
            Ramp model with the selected initial groups marked
            ``DO_NOT_USE``. For non-MIRI data or missing reference
            information, the input is returned with the RSCD step marked as
            skipped.

        Raises
        ------
        ValueError
            If either group-count override is negative.
        """
        result = self.prepare_output(step_input,
                                     open_as_type=datamodels.RampModel)

        detector = result.meta.instrument.detector
        if not detector.startswith("MIR"):
            log.warning("RSCD correction is only for MIRI data")
            log.warning("RSCD step will be skipped")
            result.meta.cal_step.rscd = "SKIPPED"
            return result

        group_skip1 = self.group_skip1
        group_skip = self.group_skip
        if group_skip1 is not None and group_skip1 < 0:
            raise ValueError("group_skip1 must be nonnegative or None")
        if group_skip is not None and group_skip < 0:
            raise ValueError("group_skip must be nonnegative or None")

        # Read CRDS values only when at least one count was not overridden.
        if group_skip1 is None or group_skip is None:
            rscd_name = self.get_reference_file(result, "rscd")
            log.info("Using RSCD reference file %s", rscd_name)

            if rscd_name == "N/A":
                log.warning("No RSCD reference file found")
                log.warning("RSCD step will be skipped")
                result.meta.cal_step.rscd = "SKIPPED"
                return result

            with datamodels.RSCDModel(rscd_name) as rscd_model:
                parameters = rscd_sub.get_rscd_parameters(
                    result, rscd_model)

            if not parameters:
                log.warning(
                    "READPATT, SUBARRAY combination not found in ref file: "
                    "RSCD correction will be skipped")
                result.meta.cal_step.rscd = "SKIPPED"
                return result

            if group_skip1 is None:
                group_skip1 = parameters["skip_int1"]
                if group_skip1 < 0:
                    log.warning("RSCD reference file is deprecated and has "
                                "no first-integration value; using 1")
                    group_skip1 = 1
            if group_skip is None:
                group_skip = parameters["skip_int2p"]

        log.info("# groups to flag in integration 1: %s", group_skip1)
        log.info("# groups to flag in integrations 2 and higher: %s",
                 group_skip)
        return rscd_sub.correction_skip_groups(
            result, group_skip1, group_skip)
